import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

if torch is not None:
    from leekrl import VecEnv, action_table
    from leekrl.model import Distribution, Inputs, Legal, Policy, gather, map_index
    from leekrl.ppo import Rollout, advantages
    from leekrl.train import (Config, evaluate, inputs, load_policy, parse, random_actions,
                              train)


def small(**changes):
    settings = dict(envs=16, steps=8, updates=3, env_threads=4, torch_threads=4, branch=16,
                    hidden=16, minibatch=64, epochs=2, eval_every=0, save_every=0,
                    deterministic=True, device="cpu")
    return Config(**{**settings, **changes})


def listed(mask):
    """Legal actions from booleans."""
    return Legal(mask.nonzero(as_tuple=True)[1], mask.sum(1))


def changed(seen, **changes):
    """The same inputs, but for some of them."""
    kept = dict(floats=seen.floats, place=seen.place, held=seen.held,
                held_sizes=seen.held_sizes, legal=seen.legal)
    return Inputs(**{**kept, **changes})


def reference_advantages(fights, gamma, lam):
    """Plain GAE on each team's list of (reward of the decision, value), per fight."""
    result = {}
    for key, decisions, terminal, bootstrap in fights:
        advantage = 0.0
        for i in range(len(decisions) - 1, -1, -1):
            reward, value = decisions[i]
            last = i == len(decisions) - 1
            following = (0.0 if terminal else bootstrap) if last else decisions[i + 1][1]
            delta = reward + gamma * following - value
            advantage = delta + (0 if last else gamma * lam * advantage)
            result[key, i] = advantage
    return result


@unittest.skipIf(torch is None, "torch is not installed")
class LearnerTests(unittest.TestCase):
    def test_advantages_of_interleaved_fights(self):
        """Compare with trajectories written separately, over random interleavings."""
        gamma, lam = 0.97, 0.9
        random = np.random.default_rng(3)
        for _ in range(200):
            steps = int(random.integers(1, 30))
            team = np.zeros((steps + 1, 1), np.uint8)
            first = np.zeros((steps + 1, 1), np.uint8)
            done = np.zeros((steps + 1, 1), np.uint8)
            rewards = np.zeros((steps + 1, 1), np.float32)
            ends = np.zeros((steps + 1, 1, 2), np.float32)
            values = random.normal(size=(steps + 1, 1)).astype(np.float32)
            fight, current = 0, {}      # (fight, team) -> positions
            received, finished = {}, set()
            for t in range(steps + 1):
                if t and random.random() < 0.15:
                    done[t] = 1
                    for k in (0, 1):
                        ends[t, 0, k] = random.normal()
                        finished.add((fight, k))
                        received[(fight, k), len(current.get((fight, k), ()))] = ends[t, 0, k]
                    fight += 1
                k = int(random.integers(2))
                team[t] = k
                positions = current.setdefault((fight, k), [])
                first[t] = not positions
                if positions:
                    rewards[t] = random.normal()
                    received[(fight, k), len(positions)] = rewards[t, 0]
                positions.append(t)
            found, valid = advantages(team[:-1], first[:-1], done[:-1], rewards[:-1],
                                      ends[:-1], values[:-1],
                                      (team[-1], first[-1], done[-1], rewards[-1], ends[-1],
                                       values[-1]), gamma, lam)
            fights, where, invalid = [], {}, set()
            for key, positions in current.items():
                # The decision after the rollout only provides a value.
                inside = [t for t in positions if t < steps]
                terminal = key in finished
                after = len(inside) < len(positions)
                bootstrap = float(values[-1, 0])
                if not terminal and not after and inside:
                    # Unknown outcome: not learned, but it still provides a value.
                    invalid.add(inside[-1])
                    bootstrap = float(values[inside[-1], 0])
                    inside = inside[:-1]
                decisions = [(float(received[key, i + 1]), float(values[t, 0]))
                             for i, t in enumerate(inside)]
                fights.append((key, decisions, terminal, bootstrap))
                for i, t in enumerate(inside):
                    where[key, i] = t
            for item, expected in reference_advantages(fights, gamma, lam).items():
                t = where[item]
                self.assertTrue(valid[t, 0])
                self.assertAlmostEqual(found[t, 0], expected, places=4)
            self.assertEqual({t for t in range(steps) if not valid[t, 0]}, invalid)

    def test_distribution(self):
        random = torch.Generator().manual_seed(1)
        mask = torch.rand(50, 200, generator=random) < 0.1
        mask[:, 0] = True
        mask[7] = False
        mask[7, 3] = True                       # a single choice
        logits = torch.randn(50, 200, generator=random) * 5
        legal = listed(mask)
        self.assertEqual(legal.rows.tolist(), mask.nonzero(as_tuple=True)[0].tolist())
        distribution = Distribution(logits[legal.rows, legal.actions], legal)
        dense = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), 1)
        torch.testing.assert_close(distribution.log_probs, dense[mask])
        entropy = -(dense.exp() * dense.masked_fill(~mask, 0)).sum(1)
        torch.testing.assert_close(distribution.entropy(), entropy)
        self.assertEqual(distribution.entropy()[7], 0)
        best = legal.actions[distribution.best()]
        torch.testing.assert_close(best, dense.argmax(1))
        counts = torch.zeros(50, 200)
        draws = 4000
        for _ in range(draws):
            position = distribution.sample(random)
            torch.testing.assert_close(legal.rows[position], torch.arange(50))
            counts[torch.arange(50), legal.actions[position]] += 1
        self.assertEqual(counts[~mask].sum(), 0)
        self.assertEqual(counts[7, 3], draws)
        self.assertLess((counts / draws - dense.exp()).abs().max(), 0.04)

    def test_gather(self):
        """Lists of some rows, taken from the lists of all of them."""
        random = np.random.default_rng(1)
        sizes = torch.from_numpy(random.integers(0, 6, 40))
        values = torch.arange(int(sizes.sum()))
        offsets = sizes.cumsum(0) - sizes
        picked = torch.from_numpy(random.permutation(40)[:15])
        found, rows = gather(values, offsets, sizes, picked, sizes[picked])
        expected = [values[offsets[p]:offsets[p] + sizes[p]] for p in picked]
        self.assertTrue(torch.equal(found, torch.cat(expected)))
        self.assertEqual(rows.tolist(), [i for i, e in enumerate(expected) for _ in e])

    def test_extreme_logits(self):
        mask = torch.ones(2, 4, dtype=torch.bool)
        legal = listed(mask)
        logits = torch.tensor([1e4, -1e4, -1e4, -1e4, -1e4, -1e4, -1e4, 1e4])
        distribution = Distribution(logits, legal)
        self.assertTrue(torch.isfinite(distribution.entropy()).all())
        for _ in range(20):
            chosen = legal.actions[distribution.sample()]
            self.assertEqual(chosen.tolist(), [0, 3])

    def test_network_sees_each_input(self):
        with VecEnv(8, threads=2, seed=3) as env:
            env.reset()
            seen = inputs(env, "cpu")
            for mask_input, head in ((True, "flat"), (False, "flat"), (True, "pair"),
                                     (False, "pair")):
                model = Policy(env.actions, env.float_count, 16, 16, mask_input, head)
                _, value = model(seen)
                self.assertEqual(value.shape, (8,))
                held = seen.held.clone()
                held[seen.held_offsets] ^= 1                # another category
                self.assertFalse(torch.equal(model(changed(seen, held=held))[1], value))
                place = (seen.place + 1) % 1225
                self.assertFalse(torch.equal(model(changed(seen, place=place))[1], value))
                floats = seen.floats.clone()
                floats[:, -100:] += 1
                self.assertFalse(torch.equal(model(changed(seen, floats=floats))[1], value))
                fewer = Legal(torch.zeros(8, dtype=torch.long), torch.ones(8))
                same = torch.equal(model(changed(seen, legal=fewer))[1], value)
                self.assertEqual(same, not mask_input)
            # The whole batch at once or row by row: same results.
            distribution, value = model(seen)
            for i in range(8):
                alone, single = model(inputs(env, "cpu", np.array([i])))
                torch.testing.assert_close(single[0], value[i])
                rows = distribution.legal.rows == i
                torch.testing.assert_close(alone.log_probs, distribution.log_probs[rows])

    def test_position_stands_for_the_outside(self):
        """The position gives back exactly which cells are outside the map."""
        outside, found = {}, set()
        random = np.random.default_rng(5)
        with VecEnv(64, threads=4, seed=9, face_enemy=False, dense_map=True) as env:
            env.reset()
            for _ in range(60):
                self.assertTrue(((env.place >= 0) & (env.place < 35 * 35)).all())
                for place, row in zip(env.place.tolist(), (env.map == 0)):
                    shape = row.tobytes()
                    self.assertEqual(outside.setdefault(place, shape), shape)
                    found.add(place)
                legal = env.legal()
                env.step((legal * random.random(legal.shape, np.float32)).argmax(1))
        self.assertGreater(len(found), 300)
        self.assertEqual(len(set(outside.values())), len(outside))

    def test_cells(self):
        """The dense map, from the list."""
        with VecEnv(16, threads=2, seed=3, dense_map=True) as env:
            env.reset()
            cells = inputs(env, "cpu").cells().numpy()
            inside = env.map > 0
            np.testing.assert_array_equal(cells[inside], env.map[inside])

    def test_pair_head(self):
        """Scores agree with the formula, written action by action."""
        with VecEnv(4, threads=2, seed=3) as env:
            env.reset()
            seen = inputs(env, "cpu")
            floats, cells = seen.floats, seen.cells()
            model = Policy(env.actions, env.float_count, 16, 16, head="pair")
            head = model.policy
            legal = seen.legal
            hidden = model._trunk(seen)
            scores = head(hidden, seen)
        table = action_table()
        blocks = floats[:, 3:3 + 18 * 116].reshape(4, 18, 116)
        seen = set()
        for n in range(0, len(scores), 7):
            row, action = int(legal.rows[n]), int(legal.actions[n])
            kind, item, dx, dy = table[action]
            tool = kind if kind < 2 else 2 + item + (37 if kind == 3 else 0)
            cell = int(map_index(torch.tensor(dx), torch.tensor(dy)))
            target = (head.category.weight[int(cells[row, cell])] + head.place.weight[cell]
                      + head.geometry(head.places)[cell])
            for block in blocks[row]:
                if block[0] and round(float(block[18] * 34)) == dx \
                        and round(float(block[19] * 34)) == dy:
                    target = target + head.entity(block)
                    seen.add((row, int(dx), int(dy)))
            wanted = (head.fixed.weight[tool]
                      + head.adapted.weight[tool] * head.weights(hidden[row])
                      + head.shared(hidden[row]))
            expected = (wanted * target).sum() + head.preference(hidden[row])[tool]
            torch.testing.assert_close(scores[n], expected, atol=1e-4, rtol=1e-4)
        self.assertTrue(seen)

    def test_action_table(self):
        table = action_table()
        self.assertEqual(table.shape, (10566, 4))
        self.assertEqual(table[0].tolist(), [0, -1, 0, 0])
        self.assertTrue((table[1:841, 0] == 1).all() and (table[841:, 0] >= 2).all())
        self.assertEqual(table[table[:, 0] == 2, 1].max(), 36)
        self.assertEqual(table[table[:, 0] == 3, 1].max(), 104)
        index = map_index(torch.from_numpy(table[:, 2]), torch.from_numpy(table[:, 3]))
        self.assertEqual(index[0], 1190)
        self.assertTrue(((index >= 0) & (index < 2381)).all())
        # Moves are all different cells.
        self.assertEqual(len(set(index[1:841].tolist())), 840)

    def test_rollout_is_faithful(self):
        """Stored log-probabilities and masks are those of the stored decisions."""
        from leekrl.train import Fights, collect
        config = small()
        with VecEnv(16, threads=4, seed=5) as env:
            env.reset()
            model = Policy(env.actions, env.float_count, 16, 16)
            rollout = Rollout(env, 8)
            collect(env, model, rollout, config, torch.Generator().manual_seed(1), Fights())
            self.assertTrue(rollout.valid.any())
            # The lists of the rollout, decision by decision.
            for _ in rollout.minibatches(1000, torch.Generator().manual_seed(1)):
                pass
            for t in range(8):
                seen, _ = rollout.pick(torch.arange(t * 16, (t + 1) * 16))
                self.assertTrue(torch.equal(seen.legal.actions, rollout.lists["legal"][t].long()))
                self.assertTrue(torch.equal(seen.held, rollout.lists["held"][t].long()))
                distribution, value = model(seen)
                position = seen.legal.offsets + rollout.choices[t]
                np.testing.assert_allclose(distribution.log_probs[position].detach(),
                                           rollout.log_probs[t], atol=1e-5)
                np.testing.assert_allclose(value.detach(), rollout.values[t], atol=1e-5)
            # Minibatches hold each decision once, with its own data.
            seen = []
            for batch, rest in rollout.minibatches(50, torch.Generator().manual_seed(2)):
                distribution, _ = model(batch)
                position = batch.legal.offsets + rest["choices"]
                np.testing.assert_allclose(distribution.log_probs[position].detach(),
                                           rest["log_probs"], atol=1e-5)
                seen += rest["advantages"].tolist()
            self.assertEqual(sorted(seen), sorted(rollout.advantages[rollout.valid].tolist()))

    def test_deterministic_training(self):
        """Two runs with the same configuration give the same metrics and weights."""
        def run(**changes):
            with tempfile.TemporaryDirectory() as folder:
                model, history = train(small(**changes), folder)
            timeless = [{k: v for k, v in metrics.items() if not k.startswith("time/")}
                        for metrics in history]
            return timeless, [p.detach().numpy().tobytes() for p in model.parameters()]
        first = run()
        self.assertEqual(first, run())
        self.assertNotEqual(first[1], run(seed=2)[1])
        # The number of environment threads changes nothing.
        self.assertEqual(first, run(env_threads=1))

    def test_scripted_opponents_are_not_learned(self):
        for opponent in ("random", "idle"):
            self.check_opponent(opponent)

    def check_opponent(self, opponent):
        from leekrl.train import Fights, collect
        config = small(opponent=opponent)
        with VecEnv(16, threads=4, seed=5) as env:
            env.reset()
            model = Policy(env.actions, env.float_count, 16, 16)
            rollout = Rollout(env, 8)
            collect(env, model, rollout, config, torch.Generator().manual_seed(1), Fights())
            other = rollout.team != np.arange(16) % 2
            self.assertFalse(rollout.valid[other].any())
            self.assertTrue(rollout.valid[~other & (rollout.team == 0)].any())
            self.assertTrue(rollout.valid[~other & (rollout.team == 1)].any())
            self.assertEqual(int(rollout.valid.sum()), sum(
                len(rest["choices"]) for _, rest in rollout.minibatches(
                    50, torch.Generator().manual_seed(1))))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "no GPU")
    def test_gpu(self):
        """Same results as on CPU, up to rounding, and the same ones each time."""
        with VecEnv(32, threads=4, seed=5) as env:
            env.reset()
            for head in ("flat", "pair"):
                model = Policy(env.actions, env.float_count, 16, 16, head=head)
                with torch.no_grad():
                    a, u = model(inputs(env, "cpu"))
                    b, v = model.cuda()(inputs(env, "cuda"))
                self.assertTrue(torch.equal(a.legal.actions, b.legal.actions.cpu()))
                torch.testing.assert_close(a.log_probs, b.log_probs.cpu(), atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(u, v.cpu(), atol=1e-4, rtol=1e-4)
                x, y = (d.sample(torch.Generator().manual_seed(1)) for d in (a, b))
                self.assertTrue(torch.equal(x, y))

        def run():
            with tempfile.TemporaryDirectory() as folder:
                model, history = train(small(device="cuda"), folder)
            return ([{k: v for k, v in m.items() if not k.startswith("time/")}
                     for m in history],
                    [p.detach().cpu().numpy().tobytes() for p in model.parameters()])
        self.assertEqual(run(), run())

    def test_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            config = small(save_every=2, updates=4)
            model, _ = train(config, folder)
            loaded, data = load_policy(Path(folder) / "checkpoint_000004.pt")
            self.assertEqual(data["update"], 4)
            self.assertEqual(data["decisions"], 4 * 8 * 16)
            for a, b in zip(model.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(a, b))
            config = small(updates=6, resume=str(Path(folder) / "checkpoint_000004.pt"))
            _, history = train(config, folder)
            self.assertEqual(len(history), 2)
            lines = (Path(folder) / "metrics.jsonl").read_text("utf8").splitlines()
            self.assertEqual(len(lines), 6)
            config = small(updates=6, hidden=8,
                           resume=str(Path(folder) / "checkpoint_000004.pt"))
            with self.assertRaisesRegex(ValueError, "another network"):
                train(config, folder)

    def test_stopping(self):
        import signal
        from leekrl import train as module
        original = module.update

        def check(ask):
            calls = []

            def update(*arguments):
                calls.append(1)
                if len(calls) == 3:
                    ask(folder)
                return original(*arguments)

            module.update = update
            try:
                _, history = train(small(updates=50, save_every=40), folder)
            finally:
                module.update = original
            self.assertEqual(len(history), 3)       # that update is finished
            self.assertTrue((Path(folder) / "checkpoint_000003.pt").exists())
            self.assertFalse((Path(folder) / "stop").exists())

        before = signal.getsignal(signal.SIGINT)
        with tempfile.TemporaryDirectory() as folder:
            check(lambda f: (Path(f) / "stop").write_text(""))
        with tempfile.TemporaryDirectory() as folder:
            check(lambda f: signal.raise_signal(signal.SIGINT))
        self.assertEqual(signal.getsignal(signal.SIGINT), before)
        # A file left by a previous run does not stop the next one.
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "stop").write_text("")
            self.assertEqual(len(train(small(updates=2), folder)[1]), 2)

    def test_evaluation(self):
        with VecEnv(1) as env:
            model = Policy(env.actions, env.float_count, 16, 16)
        results = evaluate(model, small(), (), 16, 1, ("random", "idle", model))
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertAlmostEqual(sum(result[k] for k in ("wins", "losses", "draws")), 1)
        self.assertEqual(results, evaluate(model, small(), (), 16, 1,
                                           ("random", "idle", model)))
        mirror = evaluate(model, small(), (), 16, 1, (model,), greedy=True)[0]
        self.assertAlmostEqual(sum(mirror[k] for k in ("wins", "losses", "draws")), 1)

    def test_fight_figures(self):
        from types import SimpleNamespace
        from leekrl import EPISODE_FIELDS
        from leekrl.train import Fights
        episode = np.zeros((3, len(EPISODE_FIELDS)), np.float32)
        for row, values in zip(episode, ((10, 0, 30, 50), (30, -1, 100, 140), (64, 1, 9, 9))):
            for name, value in zip(("turns", "winner", "actions_0", "actions_1"), values):
                row[EPISODE_FIELDS.index(name)] = value
        fights = Fights()
        fights.collect(SimpleNamespace(done=np.array([1, 1, 0]), episode=episode,
                                       end_rewards=np.zeros((3, 2, 6), np.float32)))
        report = fights.report()
        self.assertEqual(report["fights/finished"], 2)
        self.assertEqual(report["fights/actions"], 80)
        self.assertEqual(report["fights/actions_per_turn"], 4)

    def test_random_actions(self):
        """Uniform over the legal actions."""
        from types import SimpleNamespace
        random = np.random.default_rng(2)
        legal = random.random((6, 10566)) < 0.002
        legal[:, [0, 10565]] = True
        legal[3] = False
        legal[3, 10565] = True
        env = SimpleNamespace(legal_actions=np.concatenate([np.flatnonzero(r) for r in legal]),
                              legal_counts=legal.sum(1).astype(np.int32))
        rows = np.array([5, 0, 3, 2])
        generator = torch.Generator().manual_seed(1)
        counts = np.zeros(legal.shape)
        draws = 4000
        for _ in range(draws):
            counts[rows, random_actions(env, rows, generator)] += 1
        self.assertEqual(counts[~legal].sum(), 0)
        self.assertFalse(counts[[1, 4]].any())
        expected = (legal * draws / legal.sum(1, keepdims=True))[rows]
        error = np.abs(counts[rows] - expected) / np.sqrt(np.maximum(expected, 1))
        self.assertLess(error.max(), 5)

    def test_peek(self):
        from leekrl import peek
        names = peek.item_names()
        self.assertEqual((len(names["weapons"]), len(names["chips"]), len(names["limited"])),
                         (37, 105, 46))
        self.assertEqual(names["weapons"][0], "pistol")
        self.assertNotIn("resurrection", names["chips"])
        floats = peek.float_names(names)
        self.assertEqual(len(floats), 2705)
        self.assertEqual(floats[3 + 116 + 18], "entity 1: dx")
        self.assertEqual(floats[3 + 18 * 116], "our leek: has pistol")
        with VecEnv(1) as env:
            model = Policy(env.actions, env.float_count, 16, 16)
        fights = peek.record(model, 2, seed=4)
        self.assertEqual(json.dumps(fights), json.dumps(peek.record(model, 2, seed=4)))
        # The same world is shown, whether the network sees it turned or not.
        plain = peek.record(model, 1, seed=4, face_enemy=False)[0]["decisions"][0]
        turned = fights[0]["decisions"][0]
        self.assertTrue(turned["symmetry"] and not plain["symmetry"])
        self.assertEqual((turned["map"], turned["seen"], sorted(turned["actions"])),
                         (plain["map"], plain["seen"], sorted(plain["actions"])))
        for fight in fights:
            self.assertIsNotNone(fight["end"])
            for d in fight["decisions"]:
                self.assertIn(d["chosen"], d["actions"])
                self.assertAlmostEqual(sum(d["probs"]), 1, places=2)
                self.assertEqual(len(d["map"]), 2381)
                first, others = d["seen"]["entities"][0], d["seen"]["entities"][1:]
                self.assertEqual((first["dx"], first["dy"], first["ally"]), (0, 0, True))
                self.assertGreater(first["life"], 0)
                for e in others:    # the map agrees with the entities
                    cell = d["map"][int(map_index(torch.tensor(e["dx"]),
                                                  torch.tensor(e["dy"])))]
                    self.assertEqual(int(cell), (4 if e["ally"] else 6) + e["summon"])
            ends = {d["team"]: d["reward"] for d in fight["decisions"]}
            winner = int(fight["end"]["winner"])
            for team, reward in ends.items():
                self.assertAlmostEqual(
                    reward[3], -0.2 if winner < 0 else 1 if winner == team else -1, places=4)
        with tempfile.TemporaryDirectory() as folder:
            path = peek.write(Path(folder) / "page.html", fights, "a </script> title")
            page = path.read_text("utf8")
            self.assertNotIn("/*DATA*/null", page)
            self.assertEqual(page.count("</script>"), 1)
            train(small(updates=2, peek_every=1, peek_fights=1), folder)
            self.assertTrue((Path(folder) / "peek_000002.html").exists())

    def test_options(self):
        config = parse(["--envs", "4", "--no-mask-input", "--gamma", "0.5", "--deterministic"])
        self.assertEqual((config.envs, config.mask_input, config.gamma, config.deterministic),
                         (4, False, 0.5, True))
        self.assertEqual((config.trim, parse(["--trim", "useless"]).trim), ("harmful", "useless"))
        self.assertEqual((config.face_enemy, config.repeat_moves), (True, False))
        config = parse(["--no-face-enemy", "--repeat-moves", "--eval-opponents", "idle,random"])
        self.assertEqual((config.face_enemy, config.repeat_moves), (False, True))
        for wrong in (["--trim", "all"], ["--eval-opponents", "idle,me"], ["--layers", "0"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse(wrong)

    def test_evaluations_in_training(self):
        for opponents, expected in (("idle", {"idle"}), ("idle,random", {"idle", "random"}),
                                    ("", set())):
            with tempfile.TemporaryDirectory() as folder:
                _, history = train(small(updates=1, eval_every=1, eval_fights=4,
                                         eval_opponents=opponents), folder)
            found = {name.split("/")[1].split("_")[0] for name in history[0]
                     if name.startswith("eval/")}
            self.assertEqual(found, expected)

    def test_sight(self):
        """The sight lists reach the network, the rollout and training."""
        with VecEnv(8, threads=2, seed=3, sight=True) as env:
            env.reset()
            seen = inputs(env, "cpu")
            self.assertEqual(seen.sight.shape, (int(env.sight_counts.sum()),))
            model = Policy(env.actions, env.float_count, 16, 16, sight=True)
            _, value = model(seen)
            fewer = Inputs(seen.floats, seen.place, seen.held, seen.held_sizes, seen.legal,
                           sight=torch.zeros(8, dtype=torch.long),
                           sight_sizes=torch.ones(8, dtype=torch.long))
            self.assertFalse(torch.equal(model(fewer)[1], value))
            # The whole batch at once or row by row: same results.
            for i in range(8):
                _, single = model(inputs(env, "cpu", np.array([i])))
                torch.testing.assert_close(single[0], value[i])
            with self.assertRaisesRegex(ValueError, "sight"):
                model(changed(seen))
        from leekrl.train import Fights, collect
        with VecEnv(16, threads=4, seed=5, sight=True) as env:
            env.reset()
            model = Policy(env.actions, env.float_count, 16, 16, sight=True)
            rollout = Rollout(env, 8)
            collect(env, model, rollout, small(sight=True), torch.Generator().manual_seed(1),
                    Fights())
            for _ in rollout.minibatches(1000, torch.Generator().manual_seed(1)):
                pass
            for t in range(8):
                picked, _ = rollout.pick(torch.arange(t * 16, (t + 1) * 16))
                self.assertTrue(torch.equal(picked.sight, rollout.lists["sight"][t].long()))
                distribution, _ = model(picked)
                position = picked.legal.offsets + rollout.choices[t]
                np.testing.assert_allclose(distribution.log_probs[position].detach(),
                                           rollout.log_probs[t], atol=1e-5)
        with tempfile.TemporaryDirectory() as folder:
            trained, history = train(small(updates=2, sight=True, save_every=1, eval_every=1,
                                           eval_fights=4), folder)
            self.assertTrue(trained.settings["sight"])
            self.assertIn("eval/idle_wins", history[-1])
            loaded, _ = load_policy(Path(folder) / "checkpoint_000002.pt")
            self.assertTrue(loaded.settings["sight"])
            # A network with sight against one without, in the same fights.
            plain = Policy(trained.settings["actions"], trained.settings["floats"], 16, 16)
            results = evaluate(loaded, small(), (), 4, 1, (plain,))
            self.assertAlmostEqual(sum(results[0][k] for k in ("wins", "losses", "draws")), 1)
            with self.assertRaisesRegex(ValueError, "another network"):
                train(small(updates=3, resume=str(Path(folder) / "checkpoint_000002.pt")),
                      folder)

    def test_older_runs(self):
        """What the settings of older runs do not say."""
        from leekrl.train import world
        self.assertEqual(world({"skew": 3.0}), {
            "known_probability": 0.01, "skew": 3.0, "trim": "useless", "face_enemy": False,
            "repeat_moves": True, "sight": False})
        with tempfile.TemporaryDirectory() as folder:
            train(small(updates=1, save_every=1), folder)
            with self.assertRaisesRegex(ValueError, "turned otherwise"):
                train(small(updates=2, face_enemy=False,
                            resume=str(Path(folder) / "checkpoint_000001.pt")), folder)


if __name__ == "__main__":
    unittest.main()
