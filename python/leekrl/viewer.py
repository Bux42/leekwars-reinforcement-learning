"""Fights for the Leek Wars fight viewer: output/<id>/fight.json and logs.json.

    python -m leekrl.peek --checkpoint runs/long-1/checkpoint_010000.pt --fights 3 \\
        --viewer path/to/output

The fight data comes from the C library (see record.h). It is wrapped like a fight of
the Leek Wars API, and the logs give, for each decision, what the network thought.
"""

import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_STANDARD = 1    # AI log type of a plain message


def fight_document(data, names, winner, seed=0):
    """A fight of the Leek Wars API around the data of a recorded fight.

    names: name of each team's leek. winner: team 0 or 1, or -1 for a draw.
    """
    data = json.loads(json.dumps(data))
    leeks = [leek for leek in data["leeks"] if not leek["summon"]]
    for leek in data["leeks"]:
        leek["farmer"] = leek["team"]
        if not leek["summon"]:
            leek["name"] = names[leek["team"] - 1]
    if not data["ops"]:
        data["ops"] = {str(leek["id"]): 0 for leek in leeks}
    win = winner + 1 if winner in (0, 1) else -1
    turns = max((entry[1] for entry in data["actions"] if entry[0] == 6), default=1)

    def summary(leek):
        weapons = leek["equipment"]["weapons"]
        # The ids of the fight report are those of the scenario: team + 1 here.
        return {"id": leek["team"], "name": leek["name"], "level": leek["level"],
                "talent": 0, "hat": 0, "skin": leek["skin"], "farmer": leek["farmer"],
                "weapon": weapons[0] if weapons else 0, "title": [], "metal": 0,
                "face": leek["face"]}

    def report(leek):
        return {"id": leek["team"], "name": leek["name"],
                "dead": bool(data["dead"].get(str(leek["team"]), False)), "tb": 0, "td": 0,
                "resources": {}, "talent": 0, "talent_gain": 0, "xp": 0, "cur_xp": 0,
                "next_xp": 0, "prev_xp": 0, "level": leek["level"], "money": 0,
                "appearance": 0, "xp_locked": False}

    teams = [[leek for leek in leeks if leek["team"] == team] for team in (1, 2)]
    farmers = [{str(team): {"country": "fr", "id": team, "name": names[team - 1],
                            "avatar_changed": 0, "muted": False}} for team in (1, 2)]
    fight_id = secrets.token_hex(32)
    return {
        "id": fight_id, "date": int(time.time()), "year": datetime.now(timezone.utc).year,
        "type": 0, "context": 0, "status": 2, "winner": win,
        "leeks1": [summary(leek) for leek in teams[0]],
        "leeks2": [summary(leek) for leek in teams[1]],
        "farmers1": farmers[0], "farmers2": farmers[1],
        "report": {"duration": turns, "win": win, "season": "", "season_bonus": 0,
                   "leeks1": [report(leek) for leek in teams[0]],
                   "leeks2": [report(leek) for leek in teams[1]],
                   "bonus": 0, "flags2": []},
        "comments": [], "tournament": -1, "views": 0, "starter": 1, "trophies": [],
        "seed": int(seed), "team1_name": names[0], "team2_name": names[1], "data": data}


def actor_at(actions, index):
    """The entity whose turn it is at an action: the last turn started before it."""
    for entry in reversed(actions[:index + 1]):
        if entry[0] == 7:       # LEEK_TURN
            return entry[1]
    return 0


def fight_logs(data, messages):
    """AI logs from (index of the first action of a decision, text) pairs. Each message is
    shown at its action, as written by the entity whose turn it is."""
    logs = {}
    for index, text in messages:
        actor = actor_at(data["actions"], index)
        logs.setdefault(str(actor), {}).setdefault(str(index), []).append(
            [actor, LOG_STANDARD, text, -1, "agent", 0])
    return logs


def save(output, document, logs):
    """Writes output/<id>/fight.json and logs.json. Returns the folder."""
    folder = Path(output) / document["id"]
    folder.mkdir(parents=True, exist_ok=False)
    for name, value in (("fight.json", document), ("logs.json", logs)):
        temporary = folder / f".{name}.tmp"
        temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", "utf-8")
        os.replace(temporary, folder / name)
    return folder
