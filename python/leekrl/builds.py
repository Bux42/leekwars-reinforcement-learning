"""Known builds, written by hand in builds/known_builds.json."""

import json
from pathlib import Path

from .native import ROOT, STATS, Build, library


def _read(path):
    with open(path, encoding="utf-8-sig") as file:
        return json.load(file)


def _ids(names, known, kind, build):
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"build {build!r}: unknown {kind}: {', '.join(unknown)}")
    return [known[name] for name in names]


def load_known_builds(path=None, data=ROOT):
    """List of (name, weight, Build). Characteristics in the file include components.

    Only the format is checked here: VecEnv checks that each build is legal.
    """
    data = Path(data)
    path = Path(path) if path else data / "builds" / "known_builds.json"
    weapons = {w["name"]: w["item"] for w in _read(data / "weapons.json").values()}
    chips = {c["name"]: c["id"] for c in _read(data / "chips.json").values()}
    components = {c["name"]: c for c in _read(data / "components.json").values()}
    base = [library().leekrl_build_base(i) for i in range(len(STATS))]
    builds, names = [], set()
    for entry in _read(path)["builds"]:
        name = entry["name"]
        if name in names:
            raise ValueError(f"build {name!r} appears twice")
        names.add(name)
        totals = entry["characteristics"]
        if set(totals) != set(STATS):
            wrong = ", ".join(sorted(set(totals) ^ set(STATS)))
            raise ValueError(f"build {name!r}: missing or unknown characteristics: {wrong}")
        ids = _ids(entry["components"], {n: c["id"] for n, c in components.items()},
                   "components", name)
        for key, values, count in (("components", entry["components"], 8),
                                   ("weapons", entry["weapons"], 4)):
            if len(values) != count:
                raise ValueError(f"build {name!r}: {count} {key} expected, not {len(values)}")
        if len(entry["chips"]) > 32:
            raise ValueError(f"build {name!r}: more than 32 chips")
        build = Build()
        build.components[:] = ids
        build.weapons[:] = _ids(entry["weapons"], weapons, "weapons", name)
        for i, chip in enumerate(_ids(entry["chips"], chips, "chips", name)):
            build.chips[i] = chip
        build.n_chips = len(entry["chips"])
        for i, stat in enumerate(STATS):
            given = sum(dict(components[c]["stats"]).get(stat, 0)
                        for c in entry["components"])
            build.bought[i] = totals[stat] - base[i] - given
        builds.append((name, float(entry.get("weight", 1)), build))
    return builds
