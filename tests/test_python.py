import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from leekrl import EPISODE_FIELDS, TRIMS, WAIT, VecEnv, load_known_builds
    from leekrl.vec import take


def play(steps, seed=1, **options):
    """Play random legal actions. Returns what was seen at each step."""
    random = np.random.default_rng(0)
    seen = []
    with VecEnv(16, seed=seed, dense_map=True, **options) as env:
        env.reset()
        for _ in range(steps):
            legal = env.legal()
            actions = np.array([random.choice(np.flatnonzero(row)) for row in legal])
            seen.append((env.floats.copy(), env.map.copy(), env.legal_actions.copy(),
                         env.legal_counts.copy(), env.held.copy(), env.held_counts.copy(),
                         env.place.copy(), env.symmetry.copy(),
                         env.rewards.copy(), env.team.copy(), env.done.copy(),
                         env.episode.copy()))
            env.step(actions)
    return seen


@unittest.skipIf(np is None, "numpy is not installed")
class PythonTests(unittest.TestCase):
    def test_shapes_and_values(self):
        with VecEnv(4, threads=2) as env:
            self.assertIsNone(env.map)
        with VecEnv(4, threads=2, dense_map=True) as env:
            env.reset()
            self.assertEqual((env.actions, env.float_count), (10566, 2705))
            self.assertEqual(env.map.shape, (4, 2381))
            self.assertEqual(len(env.legal_actions), env.legal_counts.sum())
            # Both forms of the map agree.
            at = 0
            for i, n in enumerate(env.held_counts):
                held = env.held[at:at + n]
                at += n
                np.testing.assert_array_equal(np.sort(held // 6), np.flatnonzero(env.map[i] > 1))
                np.testing.assert_array_equal(env.map[i][held // 6], held % 6 + 2)
            rows = np.array([3, 0])
            legal, counts = take(env.legal_actions, env.legal_counts, rows)
            expected = [np.flatnonzero(env.legal()[r]) for r in rows]
            np.testing.assert_array_equal(legal, np.concatenate(expected))
            self.assertEqual(counts.tolist(), [len(e) for e in expected])
            self.assertTrue((env.map[:, 1190] == 3).all())
            self.assertTrue(env.legal()[:, 0].all())
            self.assertTrue(env.first.all() and not env.done.any())
            self.assertEqual(env.legal().shape, (4, 10566))
            self.assertEqual(len(env.data_hash), 64)
            # Ending every turn leads to draws after 64 turns.
            for _ in range(200):
                env.step(np.zeros(4, np.int64))
                if env.done.all():
                    break
            self.assertTrue(env.done.any())
            winner = env.episode[env.done == 1, EPISODE_FIELDS.index("winner")]
            self.assertTrue((winner == -1).all())

    def test_deterministic(self):
        first = play(60, threads=1)
        for a, b in zip(first, play(60, threads=8)):
            for x, y in zip(a, b):
                self.assertEqual(x.tobytes(), y.tobytes())
        other = play(60, seed=2)
        self.assertFalse(np.array_equal(first[-1][0], other[-1][0]))

    def test_random_builds(self):
        from leekrl import show_builds
        from leekrl.builds import describe, random_builds
        builds = random_builds(200, seed=7)
        described = [describe(b) for b in builds]
        self.assertEqual(described, [describe(b) for b in random_builds(200, seed=7)])
        self.assertNotEqual(described, [describe(b) for b in random_builds(200, seed=8)])
        # Legal, as far as the environment is concerned.
        known = [(str(i), 1, b) for i, b in enumerate(builds)]
        with VecEnv(1, known_builds=known):
            pass
        for build in described:
            self.assertEqual((len(build["components"]), len(build["weapons"])), (8, 4))
            self.assertLessEqual(len(build["chips"]), max(0, build["stats"]["ram"]))
        table = {"a": (5, 2, True, 100, 0), "b": (4, 1, False, 0, 50)}
        build = {"stats": {"tp": 15, "strength": 100, "magic": 0},
                 "weapons": ["a", "unknown"], "chips": ["b"]}
        # Twice a (11 TP, 1 of which to take it), and b.
        self.assertEqual(show_builds.best_turn(build, table), (450, (("a", 2), ("b", 1))))
        build["stats"]["tp"] = 5
        self.assertEqual(show_builds.best_turn(build, table), (50, (("b", 1),)))
        with tempfile.TemporaryDirectory() as folder:
            page = show_builds.write(Path(folder) / "page.html", show_builds.prepare(20))
            self.assertNotIn("/*DATA*/null", page.read_text("utf8"))

    def test_trims(self):
        """Each level leaves out more casts, and nothing else changes."""
        random = np.random.default_rng(3)
        envs = [VecEnv(16, seed=8, skip_forced=False, trim=trim) for trim in TRIMS]
        fewer = [0, 0]
        for env in envs:
            env.reset()
        for _ in range(300):
            masks = [env.legal() for env in envs]
            for k, (more, less) in enumerate(zip(masks, masks[1:])):
                self.assertFalse((less & ~more).any())
                np.testing.assert_array_equal(less[:, :841], more[:, :841])
                fewer[k] += int((more & ~less).sum())
            actions = [random.choice(np.flatnonzero(row)) for row in masks[-1]]
            for env in envs:
                env.step(actions)
                np.testing.assert_array_equal(env.floats, envs[0].floats)
        self.assertGreater(fewer[0], 1000)
        self.assertGreater(fewer[1], 100)
        with VecEnv(16, seed=8, skip_forced=False) as default:
            default.reset()
            envs[-1].reset()
            np.testing.assert_array_equal(default.legal(), envs[-1].legal())
        for env in envs:
            env.close()
        with self.assertRaisesRegex(ValueError, "trim must be one of"):
            VecEnv(1, trim="all")

    def test_facing(self):
        """Seen from Python: the same fights, turned."""
        from leekrl.symmetry import facing, turn, turn_back, turned_actions, turned_cells
        random = np.random.default_rng(4)
        x, y = random.integers(-30, 31, (2, 500))
        for s in range(8):
            back = turn_back(s, *turn(s, x, y))
            self.assertTrue((back[0] == x).all() and (back[1] == y).all())
        u, v = turn(facing(x, y), x, y)
        self.assertTrue(((0 <= v) & (v <= u)).all())
        actions, cells = turned_actions(), turned_cells()
        for table in (actions, cells):
            for row in table:
                self.assertEqual(len(set(row.tolist())), table.shape[1])
        options = dict(seed=6, known_builds=load_known_builds(), dense_map=True)
        rows = np.arange(32)
        with VecEnv(32, face_enemy=False, **options) as plain, VecEnv(32, **options) as env:
            plain.reset()
            env.reset()
            for _ in range(250):
                s = env.symmetry
                self.assertFalse(plain.symmetry.any())
                np.testing.assert_array_equal(env.map[rows[:, None], cells[s]], plain.map)
                legal = plain.legal()
                np.testing.assert_array_equal(env.legal()[rows[:, None], actions[s]], legal)
                chosen = np.array([random.choice(np.flatnonzero(row)) for row in legal])
                plain.step(chosen)
                env.step(actions[s, chosen])
                for name in ("rewards", "team", "first", "done", "end_rewards", "episode"):
                    np.testing.assert_array_equal(getattr(env, name), getattr(plain, name))

    def test_waiting(self):
        """An environment that waits only clears done, and others are not affected."""
        names = ("floats", "legal_counts", "held_counts", "place", "symmetry", "rewards",
                 "team", "first", "end_rewards", "episode")
        random = np.random.default_rng(1)
        with VecEnv(2, threads=2, seed=4) as env, VecEnv(2, threads=2, seed=4) as other:
            env.reset()
            other.reset()
            fights = 0
            while fights < 2:
                actions = [int(random.choice(np.flatnonzero(row))) for row in env.legal()]
                env.step(actions)
                other.step(actions)
                fights += int(env.done[0])
                kept = {name: getattr(env, name)[0].copy() for name in names}
                legal = env.legal()
                action = int(random.choice(np.flatnonzero(legal[1])))
                env.step([WAIT, action])
                self.assertEqual(env.done[0], 0)
                for name in names:
                    np.testing.assert_array_equal(getattr(env, name)[0], kept[name])
                other.step([WAIT, action])
                np.testing.assert_array_equal(env.floats, other.floats)
            with self.assertRaisesRegex(RuntimeError, "illegal action"):
                env.step([-2, 0])

    def test_records(self):
        """Recorded fights are the same, and their reports hold every action."""
        random = np.random.default_rng(2)
        reports = []
        with VecEnv(4, threads=2, seed=6) as env, VecEnv(4, threads=2, seed=6) as other:
            env.record([0, 2])
            env.reset()
            other.reset()
            self.assertEqual(env.recorded_actions(1), -1)
            while len(reports) < 6:
                actions = [int(random.choice(np.flatnonzero(row))) for row in env.legal()]
                self.assertGreater(env.recorded_actions(0), 0)
                env.step(actions)
                other.step(actions)
                np.testing.assert_array_equal(env.floats, other.floats)
                np.testing.assert_array_equal(env.episode, other.episode)
                for row in np.flatnonzero(env.done == 1):
                    report = env.report(row)
                    self.assertEqual(report is None, row % 2 == 1)
                    if report is not None:
                        reports.append((report, env.episode[row].copy()))
                self.assertIsNone(env.report(1))
        for report, episode in reports:
            self.assertEqual(sorted(report), ["actions", "dead", "leeks", "map", "ops",
                                              "version"])
            actions = report["actions"]
            self.assertEqual(actions[0], [0])                   # START_FIGHT
            turns = [a[1] for a in actions if a[0] == 6]        # NEW_TURN, from turn 2
            self.assertEqual(turns, list(range(2, len(turns) + 2)))
            # A turn can begin before the end of the fight is seen, and the turn count goes
            # past the last turn.
            self.assertIn(turns[-1] - episode[EPISODE_FIELDS.index("turns")], (-1, 0, 1))
            leeks = [leek for leek in report["leeks"] if not leek["summon"]]
            self.assertEqual(sorted(leek["team"] for leek in leeks), [1, 2])
            self.assertEqual(len({leek["id"] for leek in report["leeks"]}),
                             len(report["leeks"]))
            winner = int(episode[EPISODE_FIELDS.index("winner")])
            dead = {int(k) - 1: v for k, v in report["dead"].items()}
            self.assertEqual(sorted(dead), [0, 1])
            if winner >= 0:
                self.assertTrue(dead[1 - winner])
                self.assertFalse(dead[winner])
        self.assertGreater(max(len(r["actions"]) for r, _ in reports), 2048)

    def test_errors(self):
        with self.assertRaises(ValueError):
            VecEnv(0)
        with self.assertRaises(ValueError):
            VecEnv(1, rewards={"unknown": 1})
        with self.assertRaises(ValueError):
            VecEnv(1, stat_weights={"luck": 1})
        env = VecEnv(2)
        with self.assertRaisesRegex(RuntimeError, "not reset"):
            env.step(np.zeros(2, np.int32))
        env.reset()
        with self.assertRaises(ValueError):
            env.step(np.zeros(3, np.int32))
        illegal = np.flatnonzero(~env.legal()[1])[0]
        with self.assertRaisesRegex(RuntimeError, "environment 1.*illegal action"):
            env.step(np.array([0, illegal]))
        env.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            env.reset()

    def test_options_reach_c(self):
        # Only life: 1780 capital, hence a lot of it, and no strength.
        weights = dict.fromkeys(("strength", "wisdom", "agility", "resistance", "science",
                                 "magic", "frequency", "ram", "tp", "mp"), 0)
        with VecEnv(32, stat_weights=weights, rewards={"draw": -0.5},
                    known_probability=0) as env:
            env.reset()
            life = np.expm1(env.floats[:, 3 + 17] * 10)
            self.assertGreater(life.min(), 4000)
            while not env.done.any():
                env.step(np.zeros(32, np.int32))
            ended = env.done == 1
            np.testing.assert_allclose(env.end_rewards[ended][:, :, 3], -0.5)

    def test_known_builds(self):
        builds = load_known_builds(ROOT / "builds" / "known_builds.example.json")
        builds += load_known_builds()
        with VecEnv(8, known_builds=builds, known_probability=1) as env:
            self.assertEqual(env.known_names, [name for name, _, _ in builds])
            env.reset()
            while not env.done.any():
                env.step(np.zeros(8, np.int32))
            known = env.episode[env.done == 1][:, 8:10]
            self.assertTrue(((known >= 1) & (known <= len(builds))).all())

    def test_wrong_known_builds(self):
        source = json.loads((ROOT / "builds" / "known_builds.example.json").read_text("utf8"))

        def refused(change, message):
            data = copy.deepcopy(source)
            change(data["builds"][0])
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "builds.json"
                path.write_text(json.dumps(data), "utf8")
                with self.assertRaisesRegex(ValueError, message):
                    VecEnv(1, known_builds=load_known_builds(path)).close()

        refused(lambda b: b["weapons"].__setitem__(0, "banana"), "unknown weapons: banana")
        refused(lambda b: b["components"].pop(), "8 components expected")
        refused(lambda b: b["characteristics"].pop("mp"), "characteristics: mp")
        refused(lambda b: b["characteristics"].__setitem__("tp", 25), "capital")
        refused(lambda b: b["chips"].__setitem__(0, "resurrection"), "chip not allowed")
        refused(lambda b: b["chips"].__setitem__(0, b["chips"][1]), "duplicate chip")
        refused(lambda b: b["weapons"].__setitem__(1, "dark_katana"), "forgotten")
        refused(lambda b: b.__setitem__("name", "example_poison"), "appears twice")

    def test_import_build(self):
        """A leek as the Leek Wars API describes it becomes the same known build."""
        from leekrl.import_build import add, build_entry, leek_id

        self.assertEqual(leek_id("https://leekwars.com/leek/131073"), 131073)
        self.assertEqual(leek_id("131073"), 131073)
        with self.assertRaises(ValueError):
            leek_id("https://leekwars.com/farmer/131073")
        example = json.loads((ROOT / "builds" / "known_builds.example.json").read_text("utf-8"))
        build = example["builds"][0]

        def ids(file, key):
            with open(ROOT / file, encoding="utf-8-sig") as f:
                return {item["name"]: item for item in json.load(f).values()}, key

        tables = {kind: ids(*args) for kind, args in (
            ("components", ("components.json", "template")),
            ("weapons", ("weapons.json", "item")), ("chips", ("chips.json", "id")))}
        leek = {"id": 7, "name": "Somebody", "level": 301}
        for kind, (table, key) in tables.items():
            leek[kind] = [{"template": table[n][key], "stats": None} for n in build[kind]]
        given = {stat: sum(dict(tables["components"][0][c]["stats"]).get(stat, 0)
                           for c in build["components"])
                 for stat in build["characteristics"]}
        for stat, total in build["characteristics"].items():
            leek[stat], leek[f"total_{stat}"] = total - given[stat], total
        entry, warnings = build_entry(leek, weight=2)
        self.assertEqual(warnings, [])
        self.assertEqual(entry["name"], "Somebody")
        for key in ("characteristics", "components", "weapons", "chips"):
            self.assertEqual(entry[key], build[key])
        # Altered components keep their alterations, and fights get the leek's stats.
        leek["components"][2]["stats"] = {"life": 30, "tp": 1, "science": -20, "magic": 0}
        for stat, points in leek["components"][2]["stats"].items():
            leek[f"total_{stat}"] += points
        entry, warnings = build_entry(leek)
        self.assertEqual(warnings, [])
        self.assertEqual(entry["components"][2], {"name": build["components"][2],
                                                  "stats": {"life": 30, "tp": 1,
                                                            "science": -20}})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "known_builds.json"
            add(entry, path)
            known = load_known_builds(path)
        with VecEnv(1, threads=1, known_builds=known, known_probability=1) as env:
            env.record(0)
            env.reset()
            while not env.done[0]:
                env.step([0])
            for leek_data in env.report(0)["leeks"]:
                for stat in ("life", "tp", "science", "magic", "strength"):
                    self.assertEqual(leek_data[stat], leek[f"total_{stat}"])
        leek["chips"] = [{"template": -1}]
        with self.assertRaisesRegex(ValueError, "unknown chips"):
            build_entry(leek)
        leek["weapons"] = [None] * 4
        with self.assertRaisesRegex(ValueError, "lacks weapons"):
            build_entry(leek)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "known_builds.json"
            entry, _ = build_entry({**leek, "chips": [], "weapons": [
                {"template": tables["weapons"][0][n]["item"]} for n in build["weapons"]]})
            self.assertFalse(add(entry, path))
            with self.assertRaisesRegex(ValueError, "already named"):
                add(entry, path)
            self.assertTrue(add(entry, path, replace=True))
            self.assertEqual([n for n, _, _ in load_known_builds(path)], ["Somebody"])
            entry["characteristics"]["tp"] += 1000
            with self.assertRaises(ValueError):
                add(entry, path, replace=True)
            self.assertEqual(json.loads(path.read_text("utf-8"))["builds"][0]
                             ["characteristics"]["tp"], leek["total_tp"])


if __name__ == "__main__":
    unittest.main()
