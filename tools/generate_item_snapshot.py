#!/usr/bin/env python3
"""Generate a C item snapshot from the supplied Leek Wars JSON files."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

INT_MAX = 2**31 - 1
STATS = ("life", "strength", "wisdom", "agility", "resistance",
         "science", "magic", "tp", "mp")
# Same order as leekrl_stat in item_snapshot.h.
COMPONENT_STATS = ("life", "strength", "wisdom", "agility", "resistance", "science",
                   "magic", "frequency", "cores", "ram", "tp", "mp")


def fail(message):
    raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def integer(value, label, low=0, high=INT_MAX):
    if type(value) is not int or not low <= value <= high:
        fail(f"{label} must be an integer in [{low}, {high}]")
    return value


def text(value, label, limit=None):
    if not isinstance(value, str) or not value or "\0" in value:
        fail(f"{label} must be a nonempty string without NUL")
    raw = value.encode("utf-8")
    if limit is not None and len(raw) >= limit:
        fail(f"{label} must be shorter than {limit} UTF-8 bytes")
    return '"' + "".join(chr(c) if 32 <= c <= 126 and c not in (34, 63, 92)
                         else f"\\{c:03o}" for c in raw) + '"'


def boolean(value, label):
    if type(value) is not bool:
        fail(f"{label} must be a boolean")
    return int(value)


def number(value, label):
    if type(value) not in (int, float):
        fail(f"{label} must be a finite number")
    try:
        converted = float(value)
    except OverflowError:
        fail(f"{label} must be a finite number")
    if not math.isfinite(converted):
        fail(f"{label} must be a finite number")
    return converted.hex()


def read_catalog(path):
    data = path.read_bytes()
    obj = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique_object,
                     parse_constant=lambda value: fail(f"invalid number {value}"))
    if not isinstance(obj, dict):
        fail(f"{path}: root must be an object")
    records = []
    seen = set()
    for key, record in obj.items():
        if not isinstance(record, dict):
            fail(f"{path}: record {key} must be an object")
        ident = integer(record["id"], f"{path}: id", 1)
        if not key.isascii() or not key.isdigit() or int(key) != ident:
            fail(f"{path}: key {key!r} does not match id {ident}")
        if ident in seen:
            fail(f"{path}: duplicate record id {ident}")
        seen.add(ident)
        records.append(record)
    return sorted(records, key=lambda record: record["id"]), hashlib.sha256(data).digest()


def effect_array(effects, name, declarations, summon_refs):
    if not isinstance(effects, list) or len(effects) > 16:
        fail(f"{name}: effects must be a list of at most 16 entries")
    if not effects:
        return "NULL", 0
    values = []
    for effect in effects:
        if not isinstance(effect, dict):
            fail(f"{name}: effect must be an object")
        ident = integer(effect["id"], f"{name}: effect id", 1, 65)
        v1 = number(effect["value1"], f"{name}: value1")
        v2 = number(effect["value2"], f"{name}: value2")
        turns = integer(effect["turns"], f"{name}: turns", -1)
        targets = integer(effect["targets"], f"{name}: targets", 0, 31)
        modifiers = integer(effect["modifiers"], f"{name}: modifiers", 0, 31)
        if ident in (14, 59):
            raw_ref = effect["value1"]
            if not float(raw_ref).is_integer():
                fail(f"{name}: effect reference must be an integer")
            ref = integer(int(raw_ref), f"{name}: effect reference", 1,
                          13 if ident == 59 else INT_MAX)
            if ident == 14:
                summon_refs.add(ref)
        values.append(f"    {{{ident}, {v1}, {v2}, {turns}, {targets}, {modifiers}}}")
    declarations.append(f"static const LwEffectParameters {name}[] = {{\n" +
                        ",\n".join(values) + "\n};")
    return name, len(values)


def item_lines(records, weapon, declarations, summon_refs):
    lines, seen_ids, seen_names, seen_templates = [], set(), set(), set()
    kind = "weapon" if weapon else "chip"
    if len(records) > (256 if weapon else 512):
        fail(f"too many {kind} records")
    for index, record in enumerate(records):
        label = f"{kind} {record['id']}"
        ident = integer(record["item"] if weapon else record["id"], label + " id", 1)
        name = text(record["name"], label + " name")
        template = integer(record["template"], label + " template", 1)
        if ident in seen_ids or name in seen_names or template in seen_templates:
            fail(f"{label}: duplicate item id, name or template")
        seen_ids.add(ident)
        seen_names.add(name)
        seen_templates.add(template)
        cost = integer(record["cost"], label + " cost")
        minimum = integer(record["min_range"], label + " min_range")
        maximum = integer(record["max_range"], label + " max_range", minimum)
        launch = integer(record["launch_type"], label + " launch_type", 1, 7)
        area = integer(record["area"], label + " area", 1, 15)
        los = boolean(record["los"], label + " los")
        uses = integer(record["max_uses"], label + " max_uses", -1)
        effects, count = effect_array(record["effects"], f"{kind}_{index}_effects",
                                      declarations, summon_refs)
        common = f"{ident}, {cost}, {minimum}, {maximum}, {effects}, {count}, {launch}, {area}, {los}"
        if weapon:
            passive, n_passive = effect_array(record["passive_effects"],
                                               f"{kind}_{index}_passive", declarations,
                                               summon_refs)
            lines.append(f"    lw_weapon_init(&weapons[{index}], {common}, {template}, "
                         f"{name}, {passive}, {n_passive}, {uses});")
            lines.append(f"    if (!lw_weapons_add_weapon(&weapons[{index}])) goto rollback;")
        else:
            cooldown = integer(record["cooldown"], label + " cooldown", -1)
            team = boolean(record["team_cooldown"], label + " team_cooldown")
            initial = integer(record["initial_cooldown"], label + " initial_cooldown")
            level = integer(record["level"], label + " level", 1)
            chip_type = integer(record["type"], label + " type", 0, 9)
            lines.append(f"    lw_chip_init(&chips[{index}], {common}, {cooldown}, {team}, "
                         f"{initial}, {level}, {template}, {name}, "
                         f"(LwChipType){chip_type}, {uses});")
            lines.append(f"    if (!lw_chips_add_chip(&chips[{index}])) goto rollback;")
    return lines, seen_ids


def summon_lines(records, chip_ids):
    if len(records) > 128:
        fail("too many summons")
    lines, ids, names = [], set(), set()
    for index, record in enumerate(records):
        ident = integer(record["id"], "summon id", 1)
        name = text(record["name"], f"summon {ident} name", 64)
        if ident in ids or name in names:
            fail(f"summon {ident}: duplicate id or name")
        ids.add(ident)
        names.add(name)
        bounds = []
        for stat in STATS:
            pair = record["characteristics"][stat]
            if not isinstance(pair, list) or len(pair) != 2:
                fail(f"summon {ident}: {stat} must be [min, max]")
            minimum = integer(pair[0], stat, 1 if stat == "life" else 0)
            maximum = integer(pair[1], stat, minimum)
            bounds.extend((minimum, maximum))
        chips = record["chips"]
        states = record["states"]
        if not isinstance(chips, list) or len(chips) > 32:
            fail(f"summon {ident}: invalid chip list")
        if not isinstance(states, list):
            fail(f"summon {ident}: invalid state list")
        for chip in chips:
            integer(chip, "summon chip", 1)
            if chip not in chip_ids:
                fail(f"summon {ident}: unknown chip {chip}")
        for state in states:
            integer(state, "summon state", 1, 13)
        if len(set(chips)) != len(chips) or len(set(states)) != len(states):
            fail(f"summon {ident}: duplicate chip or state")
        zone = integer(record["zone"], "summon zone")
        lines.append(f"    lw_bulb_template_init(&summons[{index}], {ident}, {name}, " +
                     ", ".join(map(str, bounds)) + ");")
        lines.append("    initialized_summons++;")
        lines.append(f"    lw_bulb_template_set_awakening_zone(&summons[{index}], {zone});")
        for chip in chips:
            lines.append(f"    if (!lw_bulb_template_add_chip(&summons[{index}], "
                         f"lw_chips_get_chip({chip}))) goto rollback;")
        for state in states:
            lines.append(f"    if (!lw_bulb_template_add_state(&summons[{index}], {state})) goto rollback;")
        lines.append(f"    if (!lw_bulbs_add_invocation_template(&summons[{index}])) goto rollback;")
    return lines, ids


def component_rows(records):
    rows, names = [], set()
    if len(records) > 256:
        fail("too many components")
    for record in records:
        ident = integer(record["id"], "component id", 1)
        name = text(record["name"], f"component {ident} name", 64)
        if name in names:
            fail(f"component {ident}: duplicate name")
        names.add(name)
        stats = dict.fromkeys(COMPONENT_STATS, 0)
        seen = set()
        for pair in record["stats"]:
            if not isinstance(pair, list) or len(pair) != 2:
                fail(f"component {ident}: stats must be [name, value] pairs")
            stat, value = pair
            if stat not in stats or stat in seen:
                fail(f"component {ident}: unknown or duplicate stat {stat!r}")
            seen.add(stat)
            stats[stat] = integer(value, f"component {ident} {stat}", -100000, 100000)
        rows.append(f"    {{{ident}, {name}, {{{', '.join(map(str, stats.values()))}}}}}")
    return rows


def forgotten_ids(weapons):
    return [integer(weapon["item"], "weapon item", 1) for weapon in weapons
            if boolean(weapon["forgotten"], f"weapon {weapon['id']} forgotten")]


def generate(weapons, chips, summons, components, snapshot_hash):
    declarations, refs = [], set()
    components_c = component_rows(components)
    forgotten = forgotten_ids(weapons)
    weapon_lines, _ = item_lines(weapons, True, declarations, refs)
    chip_lines, chip_ids = item_lines(chips, False, declarations, refs)
    bulb_lines, bulb_ids = summon_lines(summons, chip_ids)
    if refs - bulb_ids:
        fail(f"unknown summon references: {sorted(refs - bulb_ids)}")
    checks = "\n".join(
        f"    if (lw_bulbs_get_invocation_template({ident})) "
        'return error_message(error, error_size, "summon registry is not empty");'
        for ident in sorted(bulb_ids))
    return f"""/* Generated by generate_item_snapshot.py; do not edit. */
#include "leekrl/item_snapshot.h"
#include "lw_weapon.h"
#include "lw_chip.h"
#include "lw_bulb_template.h"
#include <stdio.h>

#if LW_ATTACK_MAX_EFFECTS < 16 || LW_WEAPON_MAX_PASSIVES < 16 || LW_ENTITY_MAX_CHIPS < 32
#error Generator capacities are smaller than the snapshot generator requires.
#endif

{chr(10).join(declarations)}
static LwWeapon weapons[{max(1, len(weapons))}];
static LwChip chips[{max(1, len(chips))}];
static LwBulbTemplate summons[{max(1, len(summons))}];
static int active;
static size_t initialized_summons;
static const leekrl_component components[] = {{
{("," + chr(10)).join(components_c) or '    {0}'}
}};
static const int forgotten[] = {{{', '.join(map(str, forgotten + [0]))}}};

const leekrl_component *leekrl_components(size_t *count) {{
    if (count) *count = {len(components_c)};
    return components;
}}

int leekrl_weapon_forgotten(int item_id) {{
    size_t i;
    for (i = 0; i < {len(forgotten)}; ++i)
        if (forgotten[i] == item_id) return 1;
    return 0;
}}

const char *leekrl_item_snapshot_hash(void) {{ return "{snapshot_hash}"; }}

static int error_message(char *error, size_t size, const char *message) {{
    if (error && size) snprintf(error, size, "%s", message);
    return 0;
}}

void leekrl_item_snapshot_clear(void) {{
    if (!active) return;
    lw_bulbs_clear();
    lw_chips_clear();
    lw_weapons_clear();
    while (initialized_summons) lw_bulb_template_dispose(&summons[--initialized_summons]);
    active = 0;
}}

int leekrl_item_snapshot_register(char *error, size_t error_size) {{
    int nw, nc;
    if (error && error_size) error[0] = 0;
    if (active) return 1;
    lw_weapons_get_templates(&nw);
    lw_chips_get_templates(&nc);
    if (nw || nc) return error_message(error, error_size, "weapon/chip registries are not empty");
{checks}
    active = 1;
{chr(10).join(weapon_lines + chip_lines + bulb_lines)}
    return 1;
rollback:
    leekrl_item_snapshot_clear();
    return error_message(error, error_size, "item registry allocation failed");
}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weapons", required=True, type=Path)
    parser.add_argument("--chips", required=True, type=Path)
    parser.add_argument("--summons", required=True, type=Path)
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        weapons, whash = read_catalog(args.weapons)
        chips, chash = read_catalog(args.chips)
        summons, shash = read_catalog(args.summons)
        components, cohash = read_catalog(args.components)
        snapshot_hash = hashlib.sha256(whash + chash + shash + cohash).hexdigest()
        generated = generate(weapons, chips, summons, components, snapshot_hash)
        args.output.write_text(generated, encoding="utf-8", newline="\n")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        print(f"item snapshot generation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
