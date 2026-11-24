"""Look at the random builds fights are played with, in a local HTML page.

    python -m leekrl.show_builds --count 2000
"""

import argparse
import json
from pathlib import Path

from .builds import describe, load_known_builds, random_builds
from .native import ROOT, STATS

TEMPLATE = Path(__file__).with_name("show_builds.html")
DAMAGE, POISON, ENEMIES, STACKABLE, ON_CASTER = 1, 13, 1, 1, 4


def attacks(data=ROOT):
    """What each item can do to an enemy: cost, uses per turn, whether it is a weapon, and
    average strength damage and poison of a use, before characteristics."""
    result = {}
    for file, weapon in (("weapons.json", True), ("chips.json", False)):
        with open(Path(data) / file, encoding="utf-8-sig") as content:
            items = json.load(content).values()
        for item in items:
            direct = poison = 0
            stacks = True
            for effect in item["effects"]:
                if not effect["targets"] & ENEMIES or effect["modifiers"] & ON_CASTER:
                    continue
                value = effect["value1"] + effect["value2"] / 2
                if effect["id"] == DAMAGE:
                    direct += value
                elif effect["id"] == POISON:
                    poison += value * effect["turns"]
                    stacks = stacks and bool(effect["modifiers"] & STACKABLE)
            if direct or poison:
                uses = item["max_uses"] if item["max_uses"] > 0 else 99
                if item.get("cooldown", 0) or not stacks:
                    uses = 1
                result[item["name"]] = (item["cost"], uses, weapon, direct, poison)
    return result


def best_turn(build, known):
    """The most damage one turn can do, and the items used, with every target in reach and
    no shield, critical hit, bonus or cooldown."""
    stats, budget = build["stats"], max(0, build["stats"]["tp"])
    best = [(0.0, ())] * (budget + 1)
    for name in build["weapons"] + build["chips"]:
        if name not in known:
            continue
        cost, uses, weapon, direct, poison = known[name]
        value = (direct * (1 + max(0, stats["strength"]) / 100)
                 + poison * (1 + max(0, stats["magic"]) / 100))
        following = list(best)
        for tp in range(budget + 1):
            for n in range(1, uses + 1):
                spent = n * cost + weapon       # taking a weapon costs 1 TP
                if spent > tp:
                    break
                total = best[tp - spent][0] + n * value
                if total > following[tp][0]:
                    following[tp] = (total, best[tp - spent][1] + ((name, n),))
        best = following
    return best[budget]


def prepare(count, seed=1, skew=None, known_file=None):
    table = attacks()

    def complete(build):
        build = describe(build)
        damage, used = best_turn(build, table)
        return {**build, "damage": round(damage), "used": used}

    return {"seed": seed, "stats": STATS,
            "builds": [complete(b) for b in random_builds(count, seed, skew)],
            "known": [{"name": name, **complete(b)}
                      for name, _, b in load_known_builds(known_file)]}


def write(path, data):
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.read_text("utf8").replace("/*DATA*/null", text), "utf8")
    return path


def main():
    parser = argparse.ArgumentParser(prog="python -m leekrl.show_builds")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skew", type=float, help="that of training by default")
    parser.add_argument("--out", default=str(ROOT / "runs" / "builds.html"))
    options = parser.parse_args()
    print(write(options.out, prepare(options.count, options.seed, options.skew)))


if __name__ == "__main__":
    main()
