"""Self-play training.

    python -m leekrl.train --name first-run --updates 1000

Every setting of Config is an option. Metrics go to runs/<name>/metrics.jsonl, and to
Weights & Biases with --wandb online (or offline).
"""

import argparse
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .builds import load_known_builds
from .model import Policy
from .native import ROOT
from .ppo import Rollout, unpack, update
from .vec import EPISODE_FIELDS, REWARD_PARTS, VecEnv


@dataclass
class Config:
    name: str = "run"
    seed: int = 1
    updates: int = 1000
    # Environment
    envs: int = 256
    steps: int = 64                # decisions per environment and update
    env_threads: int = 8
    known_builds: str = ""         # file, builds/known_builds.json by default
    known_probability: float = 0.01
    skew: float = 2.0
    trim_useless: bool = True      # mask casts that cannot affect anything
    opponent: str = "self"         # self, or random or idle: not learned
    # Network
    branch: int = 128
    hidden: int = 256
    mask_input: bool = True
    head: str = "pair"             # pair (item x cell scores) or flat
    # PPO
    gamma: float = 0.999
    lam: float = 0.95
    learning_rate: float = 3e-4
    anneal: bool = False           # learning rate down to 0
    epochs: int = 4
    minibatch: int = 1024
    clip: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.0         # stop an update's epochs above it, 0 to disable
    # Run
    device: str = "cpu"
    torch_threads: int = 8
    deterministic: bool = False
    eval_every: int = 20           # updates between evaluations, 0 to disable
    eval_fights: int = 200
    save_every: int = 50
    resume: str = ""               # checkpoint to continue from
    wandb: str = "disabled"        # disabled, offline or online
    wandb_project: str = "leekwars-rl"


def seed_everything(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def inputs(env, device):
    """What the network needs, copied from the environment's current arrays."""
    return (torch.from_numpy(env.map.copy()).to(device),
            torch.from_numpy(env.floats.copy()).to(device),
            unpack(env.mask, env.actions).to(device))


def random_actions(env, generator):
    """Uniform legal actions, one per environment."""
    legal = unpack(env.mask, env.actions)
    counts = legal.sum(1)
    ends = counts.cumsum(0)
    picked = ends - counts + (torch.rand(env.envs, generator=generator, dtype=torch.float64)
                              * counts).long()
    return legal.nonzero(as_tuple=True)[1][picked].numpy()


def scripted(opponent, env, generator):
    """Actions of an opponent that does not learn: random, or idle (ends its turns)."""
    if opponent == "random":
        return random_actions(env, generator)
    return np.zeros(env.envs, np.int64)


class Fights:
    """Figures of finished fights, averaged since the last report."""

    def __init__(self):
        self.total = 0
        self.rows, self.parts = [], []

    def collect(self, env):
        over = env.done == 1
        if over.any():
            self.rows.append(env.episode[over].copy())
            self.parts.append(env.end_rewards[over].copy())
            self.total += int(over.sum())

    def report(self):
        if not self.rows:
            return {"fights/total": self.total}
        rows = np.concatenate(self.rows)
        ends = np.concatenate(self.parts)
        self.rows, self.parts = [], []
        column = {name: rows[:, i] for i, name in enumerate(EPISODE_FIELDS)}
        winner = column["winner"].astype(int)
        life = np.stack([column["life_0"], column["life_1"]], 1)
        won = winner >= 0
        result = {
            "fights/total": self.total,
            "fights/finished": len(rows),
            "fights/turns": column["turns"].mean(),
            "fights/draws": (winner == -1).mean(),
            "fights/team_0_wins": (winner == 0).mean(),
            "fights/team_1_wins": (winner == 1).mean(),
            "fights/end_life": life.mean(),
            "fights/actions": (column["actions_0"] + column["actions_1"]).mean() / 2,
            "fights/decisions": (column["decisions_0"] + column["decisions_1"]).mean() / 2,
            "fights/redrawn_maps": column["redrawn"].mean(),
        }
        if won.any():
            result["fights/winner_life"] = life[won, winner[won]].mean()
            result["fights/winner_actions"] = np.where(
                winner == 0, column["actions_0"], column["actions_1"])[won].mean()
        for i, name in enumerate(REWARD_PARTS):
            if name in ("outcome", "duration", "actions"):
                result[f"rewards/end_{name}"] = ends[:, :, i].mean()
        # How known builds fare against random ones.
        known = np.stack([column["known_0"], column["known_1"]], 1) > 0
        mixed = known[:, 0] != known[:, 1]
        if mixed.any():
            side = known[:, 1].astype(int)
            result["fights/known_vs_random_wins"] = (winner == side)[mixed].mean()
            result["fights/known_vs_random"] = int(mixed.sum())
        return {name: float(value) for name, value in result.items()}


def collect(env, model, rollout, config, generator, fights):
    """Fill the rollout. Returns the value of the decisions that follow it."""
    device = next(model.parameters()).device
    parts = np.zeros(len(REWARD_PARTS))
    for t in range(rollout.steps):
        rollout.save(t, env)
        with torch.no_grad():
            distribution, value = model(*inputs(env, device))
            position = distribution.sample(generator)
            actions = distribution.legal.actions[position].cpu().numpy()
            rollout.log_probs[t] = distribution.log_probs[position].cpu().numpy()
        rollout.values[t] = value.cpu().numpy()
        if config.opponent != "self":
            # Alternate sides: what is learned on one does not carry to the other.
            other = env.team != np.arange(env.envs) % 2
            actions = np.where(other, scripted(config.opponent, env, generator), actions)
            rollout.trained[t] = ~other
        rollout.actions[t] = actions
        env.step(actions)
        fights.collect(env)
        parts += np.abs(env.rewards).mean(0)
    with torch.no_grad():
        _, value = model(*inputs(env, device))
    rollout.finish(env, value.cpu().numpy(), config.gamma, config.lam)
    return {f"rewards/mean_abs_{name}": float(total / rollout.steps)
            for name, total in zip(REWARD_PARTS, parts) if name in ("life", "total_life",
                                                                    "summons")}


def evaluate(model, config, known, fights, seed, opponent="random", greedy=False):
    """Fights of the model against an opponent: random, idle, or another model.

    The model plays team 0 in half of the environments. Returns shares of wins, losses
    and draws among the first `fights` fights of each environment.
    """
    envs = min(128, fights)
    quota = -(-fights // envs)
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    side = np.arange(envs) % 2
    finished = np.zeros(envs, int)
    outcome = {"wins": 0, "losses": 0, "draws": 0}
    turns = []
    with VecEnv(envs, threads=config.env_threads, seed=seed, known_builds=known,
                known_probability=config.known_probability, skew=config.skew,
                trim_useless=config.trim_useless) as env:
        env.reset()
        while (finished < quota).any():
            mine = env.team == side
            with torch.no_grad():
                distribution, _ = model(*inputs(env, device))
                position = distribution.best() if greedy else distribution.sample(generator)
                actions = distribution.legal.actions[position].cpu().numpy()
                if isinstance(opponent, str):
                    others = scripted(opponent, env, generator)
                else:
                    distribution, _ = opponent(*inputs(env, device))
                    position = (distribution.best() if greedy
                                else distribution.sample(generator))
                    others = distribution.legal.actions[position].cpu().numpy()
            env.step(np.where(mine, actions, others))
            # Only the first fights count, or short ones would be over-represented.
            for i in np.flatnonzero((env.done == 1) & (finished < quota)):
                winner = int(env.episode[i, EPISODE_FIELDS.index("winner")])
                turns.append(env.episode[i, EPISODE_FIELDS.index("turns")])
                finished[i] += 1
                outcome["draws" if winner < 0 else
                        "wins" if winner == side[i] else "losses"] += 1
    total = sum(outcome.values())
    result = {name: value / total for name, value in outcome.items()}
    result["turns"] = float(np.mean(turns))
    return result


def commit():
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Log:
    """Metrics to a file, to the console, and to Weights & Biases if asked."""

    def __init__(self, config, folder, details):
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "config.json", "w", encoding="utf8") as file:
            json.dump({**asdict(config), **details}, file, indent=2)
        self.file = open(folder / "metrics.jsonl", "a" if config.resume else "w",
                         encoding="utf8")
        self.run = None
        if config.wandb != "disabled":
            import wandb
            self.run = wandb.init(project=config.wandb_project, name=config.name,
                                  mode=config.wandb, dir=str(folder),
                                  config={**asdict(config), **details})

    def write(self, step, metrics):
        self.file.write(json.dumps({"update": step, **metrics}) + "\n")
        self.file.flush()
        if self.run:
            self.run.log(metrics, step=step)
        shown = {name.split("/")[-1]: metrics[name] for name in (
            "fights/total", "fights/turns", "fights/draws", "ppo/entropy",
            "ppo/explained_variance", "time/decisions_per_second", "eval/idle_wins",
            "eval/idle_turns", "eval/random_wins") if name in metrics}
        print(f"update {step}  " + "  ".join(f"{k} {v:.3g}" for k, v in shown.items()),
              flush=True)

    def close(self):
        self.file.close()
        if self.run:
            self.run.finish()


def save(path, model, optimizer, config, step, details, decisions, fights):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "settings": model.settings,
                "optimizer": optimizer.state_dict(), "config": asdict(config),
                "update": step, "decisions": decisions, "fights": fights, **details},
               temporary)
    os.replace(temporary, path)


def load_policy(path, device="cpu"):
    """The network of a checkpoint, ready to play."""
    data = torch.load(path, map_location=device, weights_only=True)
    model = Policy(**data["settings"]).to(device)
    model.load_state_dict(data["model"])
    return model.eval(), data


def train(config, folder=None):
    """Returns the model and the metrics of each update."""
    seed_everything(config)
    known = load_known_builds(config.known_builds or None)
    folder = Path(folder) if folder else ROOT / "runs" / config.name
    device = torch.device(config.device)
    generator = torch.Generator().manual_seed(config.seed)
    env = VecEnv(config.envs, threads=config.env_threads, seed=config.seed,
                 known_builds=known, known_probability=config.known_probability,
                 skew=config.skew, trim_useless=config.trim_useless)
    details = {"data_hash": env.data_hash, "commit": commit(),
               "known_names": env.known_names}
    model = Policy(env.actions, env.float_count, config.branch, config.hidden,
                   config.mask_input, config.head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    start = decisions = 0
    fights = Fights()
    if config.resume:
        data = torch.load(config.resume, map_location=device, weights_only=True)
        if data["data_hash"] != env.data_hash or data["settings"] != model.settings:
            raise ValueError("the checkpoint was made with other game data or another network")
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        start, decisions, fights.total = data["update"], data["decisions"], data["fights"]
    log = Log(config, folder, details)
    rollout = Rollout(env, config.steps)
    history = []
    env.reset()
    try:
        for step in range(start + 1, config.updates + 1):
            if config.anneal:
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate * (1 - (step - 1) / config.updates)
            began = time.perf_counter()
            before = fights.total
            rollout.trained[:] = True
            rewards = collect(env, model, rollout, config, generator, fights)
            collected = time.perf_counter()
            result = update(model, optimizer, rollout, config, generator)
            ended = time.perf_counter()
            decisions += config.steps * config.envs
            metrics = {f"ppo/{name}": value for name, value in asdict(result).items()}
            metrics["ppo/learning_rate"] = optimizer.param_groups[0]["lr"]
            metrics.update(rewards)
            finished = fights.total - before
            metrics.update(fights.report())
            metrics.update({
                "time/decisions": decisions,
                "time/decisions_per_second": config.steps * config.envs / (ended - began),
                "time/fights_per_second": finished / (ended - began),
                "time/collect_seconds": collected - began,
                "time/update_seconds": ended - collected})
            if config.eval_every and (step % config.eval_every == 0 or
                                      step == config.updates):
                for opponent in ("idle", "random"):
                    for name, value in evaluate(model, config, known, config.eval_fights,
                                                1_000_003, opponent).items():
                        metrics[f"eval/{opponent}_{name}"] = value
            log.write(step, metrics)
            history.append(metrics)
            if config.save_every and (step % config.save_every == 0 or
                                      step == config.updates):
                save(folder / f"checkpoint_{step:06d}.pt", model, optimizer, config, step,
                     details, decisions, fights.total)
    finally:
        log.close()
        env.close()
    return model, history


def parse(arguments=None):
    parser = argparse.ArgumentParser(prog="python -m leekrl.train")
    for field in fields(Config):
        option = "--" + field.name.replace("_", "-")
        if field.type in (bool, "bool"):
            parser.add_argument(option, action=argparse.BooleanOptionalAction,
                                default=field.default)
        else:
            kind = {"int": int, "float": float, "str": str}.get(field.type, field.type)
            parser.add_argument(option, type=kind, default=field.default)
    config = Config(**vars(parser.parse_args(arguments)))
    if config.opponent not in ("self", "random", "idle"):
        parser.error("--opponent must be self, random or idle")
    if config.head not in ("pair", "flat"):
        parser.error("--head must be pair or flat")
    if config.wandb not in ("disabled", "offline", "online"):
        parser.error("--wandb must be disabled, offline or online")
    return config


if __name__ == "__main__":
    train(parse())
