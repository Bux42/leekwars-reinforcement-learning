import copy
import json
import unittest

from tools.generate_item_snapshot import generate, unique_object


def weapon():
    return {
        "id": 15, "item": 108, "name": 'sword?"',
        "cost": 5, "min_range": 1, "max_range": 1,
        "launch_type": 1, "area": 1, "los": True, "template": 15,
        "max_uses": 2, "effects": [], "passive_effects": [],
    }


class SnapshotGeneratorTests(unittest.TestCase):
    def test_weapon_uses_equipment_id_and_escapes_name(self):
        source = generate([weapon()], [], [], "hash")
        self.assertIn("lw_weapon_init(&weapons[0], 108, 5, 1, 1,", source)
        self.assertIn(r'sword\077\042', source)

    def test_missing_or_duplicate_equipment_ids_fail(self):
        bad = weapon()
        del bad["item"]
        with self.assertRaises(KeyError):
            generate([bad], [], [], "hash")
        other = copy.deepcopy(weapon())
        other.update(id=16, name="other", template=16)
        with self.assertRaisesRegex(ValueError, "duplicate item id"):
            generate([weapon(), other], [], [], "hash")

    def test_duplicate_keys_and_invalid_values_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            json.loads('{"id": 1, "id": 2}', object_pairs_hook=unique_object)
        bad = weapon()
        bad["effects"] = [{
            "id": 1, "value1": float("nan"), "value2": 1,
            "turns": 0, "targets": 31, "modifiers": 0,
        }]
        with self.assertRaisesRegex(ValueError, "finite number"):
            generate([bad], [], [], "hash")

    def test_unknown_summon_reference_fails(self):
        bad = weapon()
        bad["effects"] = [{
            "id": 14, "value1": 123.0, "value2": 0,
            "turns": 0, "targets": 31, "modifiers": 0,
        }]
        with self.assertRaisesRegex(ValueError, "unknown summon references"):
            generate([bad], [], [], "hash")


if __name__ == "__main__":
    unittest.main()
