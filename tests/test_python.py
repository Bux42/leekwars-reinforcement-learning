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
    from leekrl import EPISODE_FIELDS, VecEnv, load_known_builds


def play(steps, seed=1, **options):
    """Play random legal actions. Returns what was seen at each step."""
    random = np.random.default_rng(0)
    seen = []
    with VecEnv(16, seed=seed, **options) as env:
        env.reset()
        for _ in range(steps):
            legal = env.legal()
            actions = np.array([random.choice(np.flatnonzero(row)) for row in legal])
            seen.append((env.floats.copy(), env.map.copy(), env.mask.copy(),
                         env.rewards.copy(), env.team.copy(), env.done.copy(),
                         env.episode.copy()))
            env.step(actions)
    return seen


@unittest.skipIf(np is None, "numpy is not installed")
class PythonTests(unittest.TestCase):
    def test_shapes_and_values(self):
        with VecEnv(4, threads=2) as env:
            env.reset()
            self.assertEqual((env.actions, env.float_count, env.mask_bytes),
                             (10566, 2705, 1321))
            self.assertEqual(env.map.shape, (4, 2381))
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


if __name__ == "__main__":
    unittest.main()
