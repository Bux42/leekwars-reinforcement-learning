"""Self-play training.

    python -m leekrl.train --name first-run --updates 1000

Every setting of Config is an option. Metrics go to runs/<name>/metrics.jsonl, and to
Weights & Biases with --wandb online (or offline).

To stop early, press Ctrl+C, or create a file named "stop" in runs/<name>: the current
update is finished, and a checkpoint is saved.
"""

import argparse
import copy
import json
import os
import random
import re
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

from .builds import load_known_builds
from .model import Inputs, Legal, Policy
from .native import ROOT
from .ppo import Rollout, update
from .vec import EPISODE_FIELDS, REWARD_PARTS, TRIMS, WAIT, VecEnv, take


@dataclass
class Config:
    name: str = "run"
    seed: int = 1
    updates: int = 1000
    # Environment
    envs: int = 1024
    steps: int = 32                # decisions per environment and update
    env_threads: int = 16
    known_builds: str = ""         # file, builds/known_builds.json by default
    known_probability: float = 0.02
    skew: float = 2.0
    trim: str = "harmful"          # casts masked: nothing, useless, or harmful ones too
    face_enemy: bool = True        # turn the world so that the enemy leek is always there
    repeat_moves: bool = False     # allow moving right after a move
    sight: bool = True             # the cells from which the enemy leek can be seen
    opponent: str = "self"         # self, or random or idle: not learned
    # Network
    branch: int = 128
    hidden: int = 256
    layers: int = 2
    mask_input: bool = True
    # flat (one output per action), pair (item x cell), or target (a small network that
    # sees the situation, the item and the target cell)
    head: str = "flat"
    # PPO
    gamma: float = 0.999
    lam: float = 0.95
    learning_rate: float = 3e-4
    anneal: bool = False           # learning rate down to 0
    epochs: int = 2
    minibatch: int = 2048
    clip: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.003
    max_grad_norm: float = 0.5
    target_kl: float = 0.0         # stop an update's epochs above it, 0 to disable
    # Run
    device: str = "auto"           # auto: cuda if available, else cpu
    torch_threads: int = 8
    tf32: bool = True              # faster, slightly rounded multiplications on GPU
    deterministic: bool = False
    eval_every: int = 100          # updates between evaluations, 0 to disable
    eval_fights: int = 200
    # Separated by commas: idle, random, or networks that do not learn, as run:update
    # (runs/<run>/checkpoint_<update>.pt) or a checkpoint's path.
    eval_opponents: str = "idle"
    save_every: int = 50
    peek_every: int = 0            # updates between pages showing fights, 0 to disable
    peek_fights: int = 2
    resume: str = ""               # checkpoint to continue from
    wandb: str = "disabled"        # disabled, offline or online
    wandb_project: str = "leekwars-rl"


def world(config):
    """Settings of the environment, from those of a run, as a dictionary. Those that older
    runs do not give are what they did."""
    old = {"known_probability": 0.01, "skew": 2.0, "trim": "useless", "face_enemy": False,
           "repeat_moves": True, "sight": False}
    return {name: config.get(name, value) for name, value in old.items()}


def seed_everything(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    torch.backends.cuda.matmul.allow_tf32 = config.tf32
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def pinned(shape, dtype):
    """Arrays that are sent faster to the GPU."""
    return torch.zeros(shape, dtype=torch.from_numpy(np.zeros(1, dtype)).dtype,
                       pin_memory=True).numpy()


def allocator(device):
    return pinned if torch.device(device).type == "cuda" else np.zeros


def inputs(env, device, rows=None):
    """What the network needs, copied from the environment's current arrays."""
    sight = sight_sizes = None
    if rows is None:
        legal, sizes, held, held_sizes = (env.legal_actions, env.legal_counts, env.held,
                                          env.held_counts)
        floats, place = env.floats, env.place
        if env.sight is not None:
            sight, sight_sizes = env.sight, env.sight_counts
    else:
        legal, sizes = take(env.legal_actions, env.legal_counts, rows)
        held, held_sizes = take(env.held, env.held_counts, rows)
        floats, place = env.floats[rows], env.place[rows]
        if env.sight is not None:
            sight, sight_sizes = take(env.sight, env.sight_counts, rows)
    if sight is not None:
        sight, sight_sizes = torch.tensor(sight, device=device), torch.tensor(sight_sizes)
    return Inputs(torch.tensor(floats, device=device), torch.tensor(place, device=device),
                  torch.tensor(held, device=device), torch.tensor(held_sizes),
                  Legal(torch.tensor(legal, device=device), torch.tensor(sizes)),
                  sight=sight, sight_sizes=sight_sizes)


def random_actions(env, rows, generator):
    """A uniform legal action for some environments."""
    legal, sizes = take(env.legal_actions, env.legal_counts, rows)
    uniform = torch.rand(len(sizes), generator=generator, dtype=torch.float64).numpy()
    return legal[sizes.cumsum() - sizes + (uniform * sizes).astype(np.int64)].astype(np.int64)


def scripted(opponent, env, rows, generator):
    """Actions of an opponent that does not learn: random, or idle (ends its turns)."""
    if opponent == "random":
        return random_actions(env, rows, generator)
    return np.zeros(len(rows), np.int64)


def chosen(player, env, rows, generator, greedy=False):
    """Actions of a player, scripted or a network, for some environments."""
    if isinstance(player, str):
        return scripted(player, env, rows, generator)
    with torch.no_grad():
        seen = inputs(env, next(player.parameters()).device, rows)
        distribution, _ = player(seen)
        position = distribution.best() if greedy else distribution.sample(generator)
        legal, _ = take(env.legal_actions, env.legal_counts, rows)
        return legal[position.numpy()].astype(np.int64)


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
            # Per team, its summons included. Ending a turn is not an action.
            "fights/actions_per_turn": ((column["actions_0"] + column["actions_1"]).sum() / 2
                                        / max(column["turns"].sum(), 1)),
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
        with torch.no_grad():
            distribution, rollout.estimates[t] = model(rollout.save(t, env))
            position = distribution.sample(generator)
            rollout.log_probs[t] = distribution.log_probs[position.to(rollout.device)]
            rollout.choices[t] = (position - distribution.legal.starts).to(rollout.device)
        actions = env.legal_actions[position.numpy()].astype(np.int64)
        if config.opponent != "self":
            # Alternate sides: what is learned on one does not carry to the other.
            other = env.team != np.arange(env.envs) % 2
            rows = np.arange(env.envs)
            actions = np.where(other, scripted(config.opponent, env, rows, generator), actions)
            rollout.trained[t] = ~other
        env.step(actions)
        fights.collect(env)
        parts += np.abs(env.rewards).mean(0)
    with torch.no_grad():
        _, value = model(inputs(env, device))
    rollout.finish(env, value.cpu().numpy(), config.gamma, config.lam)
    return {f"rewards/mean_abs_{name}": float(total / rollout.steps)
            for name, total in zip(REWARD_PARTS, parts) if name in ("life", "total_life",
                                                                    "summons")}


def evaluate(model, config, known, fights, seed, opponents=("random",), greedy=False):
    """Fights of the model against each opponent: random, idle, or another model.

    The model plays team 0 in half of them. Returns, for each opponent, the shares of wins,
    losses and draws. All the fights are played side by side, one per environment:
    counting the first ones to finish would favor short ones.
    """
    envs = fights * len(opponents)
    faced = np.arange(envs) % len(opponents)
    side = np.arange(envs) // len(opponents) % 2
    generator = torch.Generator().manual_seed(seed)
    playing = np.ones(envs, bool)
    winner, turns = np.zeros(envs, int), np.zeros(envs)
    device = next(model.parameters()).device
    settings = world(asdict(config))
    # Lists that one network needs, and the others ignore.
    settings["sight"] = any(getattr(player, "settings", {}).get("sight", False)
                            for player in (model, *opponents))
    with VecEnv(envs, threads=config.env_threads, seed=seed, known_builds=known,
                allocate=allocator(device), **settings) as env:
        env.reset()
        while playing.any():
            mine = env.team == side
            actions = np.full(envs, WAIT)
            for i, player in enumerate((model, *opponents)):
                rows = np.flatnonzero(playing & (mine if i == 0 else
                                                 ~mine & (faced == i - 1)))
                if len(rows):
                    actions[rows] = chosen(player, env, rows, generator, greedy)
            env.step(actions)
            over = playing & (env.done == 1)
            winner[over] = env.episode[over, EPISODE_FIELDS.index("winner")]
            turns[over] = env.episode[over, EPISODE_FIELDS.index("turns")]
            playing &= ~over
    return [{"wins": float((winner == side)[faced == i].mean()),
             "losses": float((winner == 1 - side)[faced == i].mean()),
             "draws": float((winner == -1)[faced == i].mean()),
             "turns": float(turns[faced == i].mean())} for i in range(len(opponents))]


SCRIPTED = ("idle", "random")


def reference_path(name):
    """The checkpoint of an evaluation opponent: run:update, or a path."""
    match = re.fullmatch(r"([\w.-]+):(\d+)", name)
    if match:
        return ROOT / "runs" / match.group(1) / f"checkpoint_{int(match.group(2)):06d}.pt"
    return Path(name)


def opponent(name, device, data_hash, face_enemy):
    """An evaluation opponent: its name if scripted, else its network, which must play the
    same game in the same world."""
    if name in SCRIPTED:
        return name
    path = reference_path(name)
    if not path.is_file():
        raise ValueError(f"evaluation opponent {name!r}: no checkpoint at {path}")
    model, data = load_policy(path, device)
    if data["data_hash"] != data_hash:
        raise ValueError(f"evaluation opponent {name!r} was trained with other game data")
    if world(data["config"])["face_enemy"] != face_enemy:
        raise ValueError(f"evaluation opponent {name!r} was trained in a world turned "
                         "otherwise")
    return model


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
            # No crash report to W&B's own servers, above all when offline.
            os.environ.setdefault("WANDB_ERROR_REPORTING", "false")
            import wandb
            self.run = wandb.init(project=config.wandb_project, name=config.name,
                                  mode=config.wandb, dir=str(folder),
                                  config={**asdict(config), **details},
                                  # Neither email, computer name nor paths.
                                  settings=wandb.Settings(
                                      host="-", email="-", disable_git=True,
                                      x_disable_meta=True, x_disable_machine_info=True))

    def write(self, step, metrics):
        self.file.write(json.dumps({"update": step, **metrics}) + "\n")
        self.file.flush()
        if self.run:
            self.run.log(metrics, step=step)
        shown = {name.split("/")[-1]: metrics[name] for name in (
            "fights/total", "fights/turns", "fights/draws", "fights/actions_per_turn", "ppo/entropy",
            "ppo/explained_variance", "time/decisions_per_second", "eval/idle_wins",
            "eval/idle_turns", "eval/random_wins") if name in metrics}
        print(f"update {step}  " + "  ".join(f"{k} {v:.3g}" for k, v in shown.items()),
              flush=True)

    def close(self):
        self.file.close()
        if self.run:
            self.run.finish()


class Stopper:
    """Says when to stop: Ctrl+C, or a file named "stop" in the run's folder."""

    def __init__(self, folder):
        self.file = folder / "stop"
        self.asked = False
        self.file.unlink(missing_ok=True)

    def __enter__(self):
        # Signals can only be handled by the main thread.
        self.main = threading.current_thread() is threading.main_thread()
        if self.main:
            self.previous = signal.signal(signal.SIGINT, self._interrupt)
        return self

    def _interrupt(self, *_):
        if self.asked:
            raise KeyboardInterrupt     # the second time, at once
        self.asked = True
        print("stopping after this update (Ctrl+C again to abort)", flush=True)

    def __bool__(self):
        return self.asked or self.file.exists()

    def __exit__(self, *_):
        if self.main:
            signal.signal(signal.SIGINT, self.previous)
        self.file.unlink(missing_ok=True)


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
    if config.device == "auto":
        config = replace(config, device="cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config)
    known = load_known_builds(config.known_builds or None)
    folder = Path(folder) if folder else ROOT / "runs" / config.name
    device = torch.device(config.device)
    generator = torch.Generator().manual_seed(config.seed)
    env = VecEnv(config.envs, threads=config.env_threads, seed=config.seed,
                 known_builds=known, allocate=allocator(device), **world(asdict(config)))
    details = {"data_hash": env.data_hash, "commit": commit(),
               "known_names": env.known_names}
    model = Policy(env.actions, env.float_count, config.branch, config.hidden,
                   config.mask_input, config.head, config.layers, config.sight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5,
                                 fused=True)
    start = decisions = 0
    fights = Fights()
    if config.resume:
        data = torch.load(config.resume, map_location=device, weights_only=True)
        settings = {"layers": 2, "sight": False, **data["settings"]}  # older ones do not say
        if data["data_hash"] != env.data_hash or settings != model.settings:
            raise ValueError("the checkpoint was made with other game data or another network")
        if world(data["config"])["face_enemy"] != config.face_enemy:
            raise ValueError("the checkpoint was made with the world turned otherwise")
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        start, decisions, fights.total = data["update"], data["decisions"], data["fights"]
    log = Log(config, folder, details)
    opponents = [name for name in config.eval_opponents.split(",") if name]
    players = [opponent(name, device, env.data_hash, config.face_enemy) for name in opponents]
    rollout = Rollout(env, config.steps, device)
    history = []
    env.reset()
    try:
        with Stopper(folder) as stop:
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
                last = step == config.updates or bool(stop)
                if opponents and config.eval_every and (step % config.eval_every == 0 or last):
                    began = time.perf_counter()
                    # The same fights each time, to compare.
                    for label, result in zip(opponents, evaluate(
                            model, config, known, config.eval_fights, 1_000_003, players)):
                        for name, value in result.items():
                            metrics[f"eval/{label}_{name}"] = value
                    metrics["time/eval_seconds"] = time.perf_counter() - began
                log.write(step, metrics)
                history.append(metrics)
                if config.save_every and (step % config.save_every == 0 or last):
                    save(folder / f"checkpoint_{step:06d}.pt", model, optimizer, config, step,
                         details, decisions, fights.total)
                if config.peek_every and (step % config.peek_every == 0 or last):
                    # The same maps and builds each time, to compare.
                    from . import peek
                    model_cpu = model if config.device == "cpu" else copy.deepcopy(model).cpu()
                    peek.write(folder / f"peek_{step:06d}.html",
                               peek.record(model_cpu, config.peek_fights, 1_000_033, known,
                                           **world(asdict(config))),
                               f"{config.name}, update {step}, against itself")
                if last:
                    if step < config.updates:
                        print(f"stopped after update {step}", flush=True)
                    break
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
    if config.trim not in TRIMS:
        parser.error(f"--trim must be one of {', '.join(TRIMS)}")
    if config.layers < 1:
        parser.error("--layers must be at least 1")
    for name in filter(None, config.eval_opponents.split(",")):
        if name not in SCRIPTED and not reference_path(name).is_file():
            parser.error(f"--eval-opponents: {name!r} is neither idle, random, nor a "
                         "checkpoint (run:update or a path)")
    if config.head not in ("flat", "pair", "target"):
        parser.error("--head must be flat, pair or target")
    if config.wandb not in ("disabled", "offline", "online"):
        parser.error("--wandb must be disabled, offline or online")
    return config


if __name__ == "__main__":
    train(parse())
