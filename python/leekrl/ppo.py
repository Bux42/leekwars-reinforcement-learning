"""PPO on self-play rollouts.

Each environment interleaves the decisions of both teams. Each team of each fight is its
own trajectory: the reward and the next value of a decision come from the team's next
decision, or from the end of the fight.
"""

from dataclasses import dataclass

import numpy as np
import torch


_shifts = {}


def unpack(bits, actions):
    """Masks as booleans, from the environment's packed bits (tensor or array)."""
    if isinstance(bits, np.ndarray):
        return torch.from_numpy(
            np.unpackbits(bits, axis=1, count=actions, bitorder="little").view(bool))
    if bits.device.type == "cpu":
        return unpack(bits.numpy(), actions)
    shifts = _shifts.get(bits.device)
    if shifts is None:
        shifts = _shifts[bits.device] = torch.arange(8, dtype=torch.uint8, device=bits.device)
    flat = ((bits.unsqueeze(2) >> shifts) & 1).bool().flatten(1)
    return flat[:, :actions]


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
    """Decisions of one rollout. What the network needs stays on its device, the rest in
    arrays of shape (steps, envs, ...)."""

    def __init__(self, env, steps, device="cpu"):
        n = env.envs
        self.steps, self.envs, self.actions_count = steps, n, env.actions
        self.device = torch.device(device)
        self.floats = torch.zeros((steps, n, env.float_count), device=device)
        self.map = torch.zeros((steps, n, env.map_cells), dtype=torch.uint8, device=device)
        self.mask = torch.zeros((steps, n, env.mask_bytes), dtype=torch.uint8, device=device)
        self.actions = torch.zeros((steps, n), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((steps, n), device=device)
        self.team = np.zeros((steps, n), np.uint8)
        self.first = np.zeros((steps, n), np.uint8)
        self.done = np.zeros((steps, n), np.uint8)
        self.rewards = np.zeros((steps, n), np.float32)
        self.end_rewards = np.zeros((steps, n, 2), np.float32)
        self.values = np.zeros((steps, n), np.float32)
        self.trained = np.ones((steps, n), bool)
        self.advantages = self.returns = self.valid = None

    def save(self, t, env):
        """Keep the pending decisions, and return the inputs of the network."""
        for name in ("floats", "map", "mask"):
            getattr(self, name)[t].copy_(torch.from_numpy(getattr(env, name)))
        for name in ("team", "first", "done"):
            getattr(self, name)[t] = getattr(env, name)
        self.rewards[t] = env.rewards.sum(1)
        self.end_rewards[t] = env.end_rewards.sum(2)
        return self.map[t], self.floats[t], unpack(self.mask[t], self.actions_count)

    def finish(self, env, last_value, gamma, lam):
        last = (env.team, env.first, env.done, env.rewards.sum(1), env.end_rewards.sum(2),
                last_value)
        self.advantages, self.valid = advantages(
            self.team, self.first, self.done, self.rewards, self.end_rewards, self.values,
            last, gamma, lam)
        self.valid &= self.trained
        self.returns = self.advantages + self.values

    def minibatches(self, size, generator):
        """Shuffled minibatches of the decisions to learn from."""
        chosen = torch.from_numpy(np.flatnonzero(self.valid.reshape(-1)))
        order = torch.randperm(len(chosen), generator=generator)
        flat = {name: getattr(self, name).flatten(0, 1)
                for name in ("floats", "map", "mask", "actions", "log_probs")}
        for name in ("advantages", "returns"):
            flat[name] = torch.from_numpy(getattr(self, name).reshape(-1)).to(self.device)
        for start in range(0, len(chosen), size):
            # Reading memory in order is faster.
            picked = chosen[order[start:start + size]].sort().values.to(self.device)
            batch = {name: tensor[picked] for name, tensor in flat.items()}
            batch["mask"] = unpack(batch["mask"], self.actions_count)
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
    device = rollout.device
    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
             "grad_norm")
    # Kept on the device: reading each value would wait for the GPU.
    totals = torch.zeros(len(names), device=device)
    count = epochs = 0
    for _ in range(config.epochs):
        before = totals.clone()
        batches = 0
        for batch in rollout.minibatches(config.minibatch, generator):
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
            optimizer.step()
            with torch.no_grad():
                totals += torch.stack([
                    policy_loss, value_loss, entropy, ((ratio - 1) - log_ratio).mean(),
                    ((ratio - 1).abs() > config.clip).float().mean(), norm])
            batches += 1
            count += 1
        epochs += 1
        seen = totals - before
        if batches and not torch.isfinite(seen).all():
            raise FloatingPointError("the loss or the gradient is not finite")
        if config.target_kl and batches and seen[3].item() / batches > config.target_kl:
            break
    totals = dict(zip(names, totals.tolist()))
    returns = rollout.returns[rollout.valid]
    values = rollout.values[rollout.valid]
    variance = returns.var() if len(returns) else 0
    explained = float(1 - (returns - values).var() / variance) if variance > 0 else 0.0
    return Update(**{name: total / max(count, 1) for name, total in totals.items()},
                  explained_variance=explained, epochs=epochs, samples=len(returns))
