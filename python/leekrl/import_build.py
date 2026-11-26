"""Add the build of a public Leek Wars leek, or a loadout, to the known builds.

    python -m leekrl.import_build https://leekwars.com/leek/131073
    python -m leekrl.import_build 131073 --name my_build --weight 2
    python -m leekrl.import_build            (a loadout copied to the clipboard)
    python -m leekrl.import_build loadout.json

For a leek, asks https://leekwars.com/api/leek/get/<id>, and nothing else is sent: no
account, no cookie. A loadout is what the copy button of a fight report gives: the
capital spent on each stat, components, weapons and chips. Loadouts are read locally.
Either becomes an entry of builds/known_builds.json, checked like any known build before
the file is written.

Builds have 8 components and 4 weapons. Altered components keep their alterations.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from .builds import _read, load_known_builds
from .native import ROOT, STATS, library

API = "https://leekwars.com/api/leek/get/{}"
LEVEL = 301


def leek_id(text):
    """The id of a leek, from its id or the address of its page."""
    match = re.fullmatch(r"\s*(?:https?://(?:www\.)?leekwars\.com/leek/)?(\d+)/?\s*", text)
    if not match:
        raise ValueError(f"not a leek id or page: {text!r}")
    return int(match.group(1))


def fetch(leek):
    request = urllib.request.Request(API.format(leek),
                                     headers={"User-Agent": "leekrl-build-import"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def build_entry(leek, name=None, weight=1.0, data=ROOT):
    """A known build from the API's description of a leek. Returns (entry, warnings)."""
    def by(file, key):
        return {item[key]: item["name"] for item in _read(Path(data) / file).values()}

    def names(owned, table, kind):
        if owned is None or None in owned:
            raise ValueError(f"the leek {leek['name']!r} lacks {kind}")
        found = [table.get(item["template"]) for item in owned]
        unknown = [item["template"] for item, n in zip(owned, found) if n is None]
        if unknown:
            raise ValueError(f"unknown {kind} (templates {unknown}): is the game data "
                             "up to date?")
        return found

    warnings = []
    if leek.get("level") != LEVEL:
        warnings.append(f"level {leek.get('level')}, not {LEVEL}: its capital differs")
    components = names(leek["components"], by("components.json", "template"), "components")
    # What alterations add, or remove, as the API gives it.
    components = [{"name": n, "stats": {s: v for s, v in owned["stats"].items() if v}}
                  if owned.get("stats") and any(owned["stats"].values()) else n
                  for n, owned in zip(components, leek["components"])]
    entry = {
        "name": name or leek["name"],
        "weight": float(weight),
        "leek": leek["id"],
        "characteristics": {stat: leek[f"total_{stat}"] for stat in STATS},
        "components": components,
        "weapons": names(leek["weapons"], by("weapons.json", "item"), "weapons"),
        "chips": names(leek["chips"], by("chips.json", "id"), "chips"),
    }
    return entry, warnings


def loadout_entry(loadout, name=None, weight=1.0, data=ROOT):
    """A known build from a loadout, at level 301. Returns (entry, warnings)."""
    if loadout.get("type") != "leek-wars-loadout":
        raise ValueError("not a Leek Wars loadout")
    if loadout.get("version") != 1:
        raise ValueError(f"loadout version {loadout.get('version')}: only 1 is known")
    lib = library()
    spent = loadout.get("stats", {})
    unknown = set(spent) - set(STATS)
    if unknown:
        raise ValueError(f"unknown stats: {', '.join(sorted(unknown))}")
    standard = {c["name"]: dict(c["stats"])
                for c in _read(Path(data) / "components.json").values()}
    components, unknown = [], []
    for slot in sorted(loadout.get("components", []), key=lambda c: c["slot"]):
        if slot["item"] not in standard:
            unknown.append(slot["item"])
        altered = {s: v for s, v in (slot.get("stats") or {}).items() if v}
        components.append({"name": slot["item"], "stats": altered} if altered
                          else slot["item"])
    if unknown:
        raise ValueError(f"unknown components: {', '.join(unknown)}: is the game data up to "
                         "date?")
    characteristics = {}
    for index, stat in enumerate(STATS):
        points = lib.leekrl_build_points(index, int(spent.get(stat, 0)))
        if points < 0:
            raise ValueError(f"{spent[stat]} capital cannot be spent exactly on {stat}")
        given = 0
        for component in components:
            if isinstance(component, str):
                component = {"name": component, "stats": {}}
            given += standard[component["name"]].get(stat, 0) + component["stats"].get(stat, 0)
        characteristics[stat] = lib.leekrl_build_base(index) + points + given
    entry = {
        "name": name or loadout.get("name") or "loadout",
        "weight": float(weight),
        "characteristics": characteristics,
        "components": components,
        "weapons": [w.removeprefix("weapon_") for w in
                    loadout.get("weapons", []) + loadout.get("forgotten_weapons", [])],
        "chips": [c.removeprefix("chip_") for c in loadout.get("chips", [])],
    }
    return entry, []


def clipboard():
    """The text of the clipboard."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        try:
            return root.clipboard_get()
        finally:
            root.destroy()
    except Exception:
        pass
    if sys.platform == "win32":
        return subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                              capture_output=True, text=True, encoding="utf-8",
                              check=True).stdout
    raise ValueError("cannot read the clipboard: give the loadout as a file")


def parse_loadout(text, where):
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        hint = ""
        if where == "argument" and '"' not in text:
            hint = (": the shell removed its quotes. Copy the loadout and run the command "
                    "without argument, to read the clipboard")
        raise ValueError(f"the loadout of the {where} is not valid JSON{hint or f': {error}'}"
                         ) from error


def add(entry, path, replace=False, data=ROOT):
    """Adds an entry to a known builds file, once checked. Returns whether it replaced one."""
    from .vec import VecEnv

    path = Path(path)
    content = _read(path) if path.exists() else {"builds": []}
    builds = content["builds"]
    same = [i for i, b in enumerate(builds) if b["name"] == entry["name"]]
    if same and not replace:
        raise ValueError(f"a build is already named {entry['name']!r}: give another --name, "
                         "or --replace it")
    if same:
        builds[same[0]] = entry
    else:
        builds.append(entry)
    text = json.dumps(content, indent=2, ensure_ascii=False) + "\n"
    # Checked as training would read it, before the file changes.
    with tempfile.TemporaryDirectory() as folder:
        trial = Path(folder) / "known_builds.json"
        trial.write_text(text, "utf-8")
        known = load_known_builds(trial, data)
        VecEnv(1, threads=1, known_builds=known).close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")
    return bool(same)


def main(arguments=None):
    parser = argparse.ArgumentParser(prog="python -m leekrl.import_build")
    parser.add_argument("source", nargs="?",
                        help="id of a leek, or the address of its page; or a loadout file, "
                             "or its JSON text. The clipboard's loadout if not given")
    parser.add_argument("--name", help="name of the build, the leek's or the loadout's by "
                                       "default")
    parser.add_argument("--weight", type=float, default=1.0,
                        help="how often it is drawn, among the known builds")
    parser.add_argument("--file", default=ROOT / "builds" / "known_builds.json")
    parser.add_argument("--replace", action="store_true",
                        help="replace the build of the same name")
    options = parser.parse_args(arguments)
    if not options.weight > 0:
        parser.error("the weight must be positive")
    try:
        source = options.source
        if source is None:
            loadout = parse_loadout(clipboard(), "clipboard")
        elif source.lstrip().startswith("{"):
            loadout = parse_loadout(source, "argument")
        elif Path(source).is_file():
            loadout = parse_loadout(Path(source).read_text("utf-8-sig"), "file")
        else:
            loadout = None
        if loadout is not None:
            entry, warnings = loadout_entry(loadout, options.name, options.weight)
        else:
            leek = fetch(leek_id(options.source))
            entry, warnings = build_entry(leek, options.name, options.weight)
        replaced = add(entry, options.file, options.replace)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        sys.exit(f"error: {error}")
    for warning in warnings:
        print(f"warning: {warning}")
    print(f"{'replaced' if replaced else 'added'} {entry['name']!r} in {options.file}: "
          f"{len(entry['weapons'])} weapons, {len(entry['chips'])} chips, "
          + ", ".join(f"{s} {v}" for s, v in entry["characteristics"].items()))


if __name__ == "__main__":
    main()
