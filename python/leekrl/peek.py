"""Look at what the agent sees and does, in a local HTML page.

    python -m leekrl.peek --checkpoint runs/first-run/checkpoint_000100.pt --fights 3

Everything shown about the world is decoded from the inputs of the network, and from
nothing else: what is missing from the page is missing for the agent.
"""

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch

from .builds import load_known_builds
from .model import Policy
from .native import ROOT, library
from .vec import EPISODE_FIELDS, REWARD_PARTS, TRIMS, VecEnv, action_table

TEMPLATE = Path(__file__).with_name("peek.html")

# Same orders and scales as specs/Observation.md.
STATES = ("unhealable", "dense", "rooted", "sterile", "amputated")
EFFECTS = {
    2: ("heal over time", 500), 3: ("strength", 500), 4: ("agility", 500),
    5: ("relative shield", 100), 6: ("absolute shield", 500), 7: ("MP", 10), 8: ("TP", 10),
    13: ("poison", 500), 17: ("MP shackle", 10), 18: ("TP shackle", 10),
    19: ("strength shackle", 500), 20: ("damage return", 100), 21: ("resistance", 500),
    22: ("wisdom", 500), 25: ("aftereffect", 500), 26: ("vulnerability", 100),
    27: ("absolute vulnerability", 500), 29: ("stolen shield", 500), 31: ("raw MP", 10),
    32: ("raw TP", 10), 37: ("raw absolute shield", 500), 38: ("raw strength", 500),
    39: ("raw magic", 500), 40: ("raw science", 500), 41: ("raw agility", 500),
    42: ("raw resistance", 500), 44: ("raw wisdom", 500), 47: ("agility shackle", 500),
    48: ("wisdom shackle", 500), 52: ("raw power", 100), 54: ("raw relative shield", 100)}
STATS = (("strength", 1000), ("wisdom", 1000), ("agility", 1000), ("resistance", 1000),
         ("science", 1000), ("magic", 1000), ("frequency", 1000), ("power", 100),
         ("damage return", 100), ("absolute shield", 500), ("relative shield", 100))
ENTITY, ENTITIES = 116, 18


def item_names(data=ROOT):
    """Names of the weapons and chips of the observation, in order, and of summons."""
    def read(name):
        with open(Path(data) / name, encoding="utf-8-sig") as file:
            return json.load(file).values()

    table = action_table()
    weapons = np.zeros(table[table[:, 0] == 2, 1].max() + 1, np.int32)
    chips = np.zeros(table[table[:, 0] == 3, 1].max() + 1, np.int32)
    pointer = ctypes.POINTER(ctypes.c_int32)
    if not library().leekrl_vec_items(weapons.ctypes.data_as(pointer),
                                      chips.ctypes.data_as(pointer)):
        raise RuntimeError("cannot list the items")
    weapon_data = {w["item"]: w for w in read("weapons.json")}
    chip_data = {c["id"]: c for c in read("chips.json")}
    summons = {s["id"]: s for s in read("summons.json")}
    limited = ([weapon_data[i]["name"] for i in weapons
                if weapon_data[i]["max_uses"] != -1]
               + [chip_data[i]["name"] for i in chips if chip_data[i]["max_uses"] != -1])
    return {"weapons": [weapon_data[i]["name"] for i in weapons],
            "chips": [chip_data[i]["name"] for i in chips], "limited": limited,
            "types": ["leek"] + [summons[i]["name"] for i in range(1, 14)],
            "summon_chips": [[]] + [[chip_data[c]["name"] for c in summons[i]["chips"]]
                                    for i in range(1, 14)]}


def float_names(names):
    """A name for each float of the observation."""
    result = ["turn", "ally summons", "enemy summons"]
    block = ["present", "ally"] + [f"type {t}" for t in names["types"]]
    block += ["life / total", "log total life", "dx", "dy", "distance", "TP", "total TP",
              "MP", "total MP"]
    block += [name for name, _ in STATS]
    for state in STATES:
        block += [f"state {state}", f"state {state} duration"]
    for name, _ in EFFECTS.values():
        block += [f"effect {name}", f"effect {name} duration"]
    block += ["awakening zone", "already awakened"]
    block += [f"summon chip {i + 1} cooldown" for i in range(6)]
    assert len(block) == ENTITY
    for i in range(ENTITIES):
        result += [f"entity {i}: {name}" for name in block]
    for side in ("our leek", "enemy leek"):
        result += [f"{side}: has {n}" for n in names["weapons"] + names["chips"]]
        result += [f"{side}: cooldown {n}" for n in names["chips"]]
        result += [f"{side}: holds {n}" for n in names["weapons"]]
    return result + [f"uses left: {n}" for n in names["limited"]]


def turns(value):
    """A duration, from its scaled value: turns left, or "permanent"."""
    return "permanent" if value > 0.95 else "9+" if value > 0.89 else round(float(value) * 10)


def decode(floats, names):
    """The observation's floats in readable units."""
    entities = []
    for i in range(ENTITIES):
        b = floats[3 + i * ENTITY:3 + (i + 1) * ENTITY]
        if b[0] == 0:
            continue
        kind = int(np.argmax(b[2:16]))
        total = float(np.expm1(b[17] * 10))
        entity = {
            "type": names["types"][kind], "ally": bool(b[1]), "summon": kind > 0,
            "life": round(float(b[16]) * total), "total_life": round(total),
            "dx": round(float(b[18]) * 34), "dy": round(float(b[19]) * 34),
            "tp": round(float(b[21]) * 30), "total_tp": round(float(b[22]) * 30),
            "mp": round(float(b[23]) * 15), "total_mp": round(float(b[24]) * 15),
            "stats": {name: round(float(b[25 + k]) * scale)
                      for k, (name, scale) in enumerate(STATS)},
            "states": [[name, turns(b[37 + 2 * k])] for k, name in enumerate(STATES)
                       if b[36 + 2 * k]],
            "effects": [[name, round(float(b[46 + 2 * k]) * scale)
                         if b[46 + 2 * k] < 5 else f"{5 * scale}+", turns(b[47 + 2 * k])]
                        for k, (name, scale) in enumerate(EFFECTS.values())
                        if b[46 + 2 * k] or b[47 + 2 * k]],
            "zone": round(float(b[108]) * 10), "awakened": bool(b[109]),
            "cooldowns": [[name, round(float(b[110 + k]) * 10)]
                          for k, name in enumerate(names["summon_chips"][kind])]}
        entities.append(entity)
    weapons, chips = len(names["weapons"]), len(names["chips"])
    size = 2 * weapons + 2 * chips
    start = 3 + ENTITIES * ENTITY
    leeks = []
    for side in range(2):
        b = floats[start + side * size:start + (side + 1) * size]
        held = np.flatnonzero(b[weapons + 2 * chips:])
        leeks.append({
            "weapons": [names["weapons"][k] for k in np.flatnonzero(b[:weapons])],
            "chips": [[names["chips"][k], round(float(b[weapons + chips + k]) * 10)]
                      for k in np.flatnonzero(b[weapons:weapons + chips])],
            "holds": names["weapons"][held[0]] if len(held) else None})
    uses = floats[start + 2 * size:]
    return {"turn": round(float(floats[0]) * 64), "entities": entities, "leeks": leeks,
            "uses": [[names["limited"][k], round(float(uses[k]) * 5)]
                     for k in np.flatnonzero(uses)]}


def record(model, fights, seed, known=(), opponent="self", max_decisions=3000, **settings):
    """Play fights, keeping every decision. opponent: self, random or idle (team 1)."""
    names = item_names()
    result = []
    generator = torch.Generator().manual_seed(seed)
    for number in range(fights):
        with VecEnv(1, threads=1, seed=seed + number, known_builds=known, **settings) as env:
            env.reset()
            decisions, last = [], [None, None]
            while len(decisions) < max_decisions:
                team = int(env.team[0])
                floats, cells = env.floats[0].copy(), env.map[0].copy()
                mask = torch.from_numpy(env.legal())
                with torch.no_grad():
                    distribution, value = model(torch.from_numpy(cells[None]),
                                                torch.from_numpy(floats[None]), mask)
                    actions = distribution.legal.actions
                    if opponent == "idle" and team == 1:
                        position = 0
                    elif opponent == "random" and team == 1:
                        position = int(torch.randint(len(actions), (1,), generator=generator))
                    else:
                        position = int(distribution.sample(generator)[0])
                if last[team] is not None:
                    decisions[last[team]]["reward"] = env.rewards[0].round(4).tolist()
                last[team] = len(decisions)
                decisions.append({
                    "team": team, "value": round(float(value[0]), 4),
                    "entropy": round(float(distribution.entropy()[0]), 3),
                    "scripted": opponent != "self" and team == 1,
                    "map": "".join(map(str, cells)),
                    "seen": decode(floats, names),
                    "raw": [[int(k), round(float(floats[k]), 4)]
                            for k in np.flatnonzero(floats)],
                    "actions": actions.tolist(),
                    "probs": [round(float(p), 5) for p in distribution.probs],
                    "chosen": int(actions[position])})
                env.step(np.array([int(actions[position])]))
                if env.done[0]:
                    break
            end = None
            if env.done[0]:
                for team in (0, 1):
                    if last[team] is not None:
                        decisions[last[team]]["reward"] = (
                            env.end_rewards[0, team].round(4).tolist())
                end = dict(zip(EPISODE_FIELDS, env.episode[0].round(4).tolist()))
            result.append({"seed": seed + number, "decisions": decisions, "end": end})
    return result


def write(path, fights, title):
    names = item_names()
    data = {"title": title, "fights": fights, "parts": REWARD_PARTS,
            "weapons": names["weapons"], "chips": names["chips"],
            "table": action_table().reshape(-1).tolist(), "floats": float_names(names)}
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text("utf8").replace("/*DATA*/null", text)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, "utf8")
    return path


def main():
    parser = argparse.ArgumentParser(prog="python -m leekrl.peek")
    parser.add_argument("--checkpoint", help="an untrained network if not given")
    parser.add_argument("--fights", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--opponent", choices=("self", "random", "idle"), default="self")
    parser.add_argument("--known-probability", type=float, default=0.01)
    parser.add_argument("--trim", choices=TRIMS,
                        help="casts masked, as in training by default")
    parser.add_argument("--out", help="runs/peek.html, or next to the checkpoint")
    options = parser.parse_args()
    if options.checkpoint:
        from .train import load_policy
        model, data = load_policy(options.checkpoint)
        title = f"{data['config']['name']}, update {data['update']}"
        # Runs that do not say only masked useless casts.
        trim = options.trim or data["config"].get("trim", "useless")
        out = options.out or Path(options.checkpoint).with_suffix(".html")
    else:
        torch.manual_seed(options.seed)
        with VecEnv(1) as env:
            model = Policy(env.actions, env.float_count)
        title, out = "untrained network", options.out or ROOT / "runs" / "peek.html"
        trim = options.trim
    fights = record(model, options.fights, options.seed, load_known_builds(),
                    options.opponent, known_probability=options.known_probability,
                    trim=trim)
    print(write(out, fights, f"{title}, against {options.opponent}"))


if __name__ == "__main__":
    main()
