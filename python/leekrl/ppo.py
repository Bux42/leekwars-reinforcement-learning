"""PPO on self-play rollouts.

Each environment interleaves the decisions of both teams. Each team of each fight is its
own trajectory: the reward and the next value of a decision come from the team's next
decision, or from the end of the fight.
"""

from dataclasses import dataclass

import numpy as np
import torch


def unpack(bits, actions):
    """Masks as booleans, from the environment's packed bits."""
    return torch.from_numpy(
        np.unpackbits(bits, axis=1, count=actions, bitorder="little").view(bool))


def advantages(team, first, done, rewards, end_rewards, values, last, gamma, lam):
    """Generalized advantage estimation for interleaved trajectories.

    Arrays of shape (steps, envs): at each decision, the deciding team, whether it is its
    first one in the fight, whether a fight ended just before, the reward of the team's
    previous decision, and the value. end_rewards is (steps, envs, 2). last holds the same
    for the decision following the rollout, as a tuple.

    Returns advantages, and whether they are valid: the last decision of a team in the
    rollout is not, when its outcome is still unknown.
    """
    steps, envs = team.shape
    index = np.arange(envs)
    known = np.zeros((envs, 2), bool)
    reward = np.zeros((envs, 2), np.float32)
    next_value = np.zeros((envs, 2), np.float32)
    next_advantage = np.zeros((envs, 2), np.float32)
    alive = np.zeros((envs, 2), np.float32)

    def precede(team, first, done, received, ended, value, advantage):
        """Prepare what earlier decisions will see."""
        team = team.astype(np.intp)
        known[index, team] = first == 0
        reward[index, team] = received
        next_value[index, team] = value
        next_advantage[index, team] = advantage
        alive[index, team] = 1
        over = done != 0
        known[over] = True
        reward[over] = ended[over]
        next_value[over] = 0
        next_advantage[over] = 0
        alive[over] = 0

    precede(*last, np.zeros(envs, np.float32))
    result = np.zeros((steps, envs), np.float32)
    valid = np.zeros((steps, envs), bool)
    for t in range(steps - 1, -1, -1):
        k = team[t].astype(np.intp)
        usable = known[index, k]
        delta = (reward[index, k] + gamma * alive[index, k] * next_value[index, k]
                 - values[t])
        current = np.where(
            usable, delta + gamma * lam * alive[index, k] * next_advantage[index, k], 0)
        result[t] = current
        valid[t] = usable
        precede(team[t], first[t], done[t], rewards[t], end_rewards[t], values[t], current)
    return result, valid


class Rollout:
    """Decisions of one rollout, as arrays of shape (steps, envs, ...)."""

    def __init__(self, env, steps):
        n = env.envs
        self.steps, self.envs, self.actions_count = steps, n, env.actions
        self.floats = np.zeros((steps, n, env.float_count), np.float32)
        self.map = np.zeros((steps, n, env.map_cells), np.uint8)
        self.mask = np.zeros((steps, n, env.mask_bytes), np.uint8)
        self.team = np.zeros((steps, n), np.uint8)
        self.first = np.zeros((steps, n), np.uint8)
        self.done = np.zeros((steps, n), np.uint8)
        self.rewards = np.zeros((steps, n), np.float32)
        self.end_rewards = np.zeros((steps, n, 2), np.float32)
        self.actions = np.zeros((steps, n), np.int64)
        self.log_probs = np.zeros((steps, n), np.float32)
        self.values = np.zeros((steps, n), np.float32)
        self.trained = np.ones((steps, n), bool)
        self.advantages = self.returns = self.valid = None

    def save(self, t, env):
        for name in ("floats", "map", "mask", "team", "first", "done"):
            getattr(self, name)[t] = getattr(env, name)
        self.rewards[t] = env.rewards.sum(1)
        self.end_rewards[t] = env.end_rewards.sum(2)

    def finish(self, env, last_value, gamma, lam):
        last = (env.team, env.first, env.done, env.rewards.sum(1), env.end_rewards.sum(2),
                last_value)
        self.advantages, self.valid = advantages(
            self.team, self.first, self.done, self.rewards, self.end_rewards, self.values,
            last, gamma, lam)
        self.valid &= self.trained
        self.returns = self.advantages + self.values

    def minibatches(self, size, generator, device):
        """Shuffled minibatches of the decisions to learn from."""
        chosen = np.flatnonzero(self.valid.reshape(-1))
        order = torch.randperm(len(chosen), generator=generator).numpy()
        flat = {name: getattr(self, name).reshape((-1,) + getattr(self, name).shape[2:])
                for name in ("floats", "map", "mask", "actions", "log_probs", "values",
                             "advantages", "returns")}
        for start in range(0, len(chosen), size):
            # Fancy indexing on sorted positions is faster.
            picked = np.sort(chosen[order[start:start + size]])
            batch = {name: torch.from_numpy(array[picked]).to(device)
                     for name, array in flat.items() if name != "mask"}
            batch["mask"] = unpack(flat["mask"][picked], self.actions_count).to(device)
            yield batch


@dataclass
class Update:
    """Averages over the minibatches of an update."""
    policy_loss: float = 0
    value_loss: float = 0
    entropy: float = 0
    approx_kl: float = 0
    clip_fraction: float = 0
    grad_norm: float = 0
    explained_variance: float = 0
    epochs: int = 0
    samples: int = 0


def update(model, optimizer, rollout, config, generator):
    device = next(model.parameters()).device
    totals = {name: 0.0 for name in ("policy_loss", "value_loss", "entropy", "approx_kl",
                                     "clip_fraction", "grad_norm")}
    count = epochs = 0
    for _ in range(config.epochs):
        kl = batches = 0
        for batch in rollout.minibatches(config.minibatch, generator, device):
            distribution, value = model(batch["map"], batch["floats"], batch["mask"])
            log_prob = distribution.log_probs[distribution.find(batch["actions"])]
            log_ratio = log_prob - batch["log_probs"]
            ratio = log_ratio.exp()
            advantage = batch["advantages"]
            if len(advantage) > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            clipped = ratio.clamp(1 - config.clip, 1 + config.clip)
            policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
            value_loss = 0.5 * (value - batch["returns"]).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = (policy_loss + config.value_weight * value_loss
                    - config.entropy_weight * entropy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError("the gradient is not finite")
            optimizer.step()
            with torch.no_grad():
                batch_kl = ((ratio - 1) - log_ratio).mean().item()
                totals["clip_fraction"] += (
                    (ratio - 1).abs() > config.clip).float().mean().item()
            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy.item()
            totals["approx_kl"] += batch_kl
            totals["grad_norm"] += norm.item()
            kl += batch_kl
            batches += 1
            count += 1
        epochs += 1
        if config.target_kl and batches and kl / batches > config.target_kl:
            break
    returns = rollout.returns[rollout.valid]
    values = rollout.values[rollout.valid]
    variance = returns.var() if len(returns) else 0
    explained = float(1 - (returns - values).var() / variance) if variance > 0 else 0.0
    return Update(**{name: total / max(count, 1) for name, total in totals.items()},
                  explained_variance=explained, epochs=epochs, samples=len(returns))
