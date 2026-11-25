# Leek Wars Reinforcement Learning

C project using `leekwars-generator-c` as a Git submodule.

Clone with the submodule:

```sh
git clone --recurse-submodules https://github.com/Bux42/leekwars-reinforcement-learning.git
```

For an existing checkout, initialize it with:

```sh
git submodule update --init --recursive
```

The submodule's `rl` branch holds the RL-only changes and starts from
`java-parity`. Pure parity fixes belong on `java-parity`, then get merged into
`rl`. Normal checkouts use the commit recorded by this repository.

## Build and test

Requires a C99 compiler, CMake 3.20 or newer, and an initialized generator
submodule.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Environment interface

`leekrl/env.h` defines a Gymnasium-style interface for environments with
fixed-size float observations and discrete actions. An environment supplies
reset and step callbacks; the wrapper validates observations, rewards and
action masks and rejects illegal actions. Reset takes an explicit seed.

Each successful step reports reward, `terminated` and `truncated` separately.
The final observation remains available after either ending so a learner can
bootstrap a truncation that is not also a termination. A completed or failed
environment must be reset before another step. The environment borrows
callback context; the caller owns it.

### Leek Wars starter scenario

`leekrl/leekwars.h` supplies a generator-backed 1v1 environment against an idle
opponent. Each learner turn chooses wait, approach or fire. Observations
contain turn, health, cells, TP and MP; they are raw values, not the full game
state. Register a supported weapon with the generator before creating the
environment. See `tests/test_leekwars_training.c` for a working setup.

Wins and losses reward +1 and -1; a turn-limit draw rewards 0 and terminates
the episode. The adapter replays the complete fight at every step because the
generator currently runs fights synchronously. This is a correctness-first
prototype, not a high-throughput implementation. It uses only the low 31 bits
of each seed and does not support concurrent generator use or registry changes.

### Fixed actions and masks

`leekrl/actions.h` provides reusable action descriptors for movement
destinations, weapon casts, chip casts and ending a turn. Targets are relative
to the acting entity in generator coordinates. `leekrl_action_cast_layout` builds a
deterministically ordered list of range/launch-type offsets from an attack;
for example, an inline range of 1–6 yields 24 slots. Add movement and end-turn
slots separately.

At each decision, `leekrl_action_catalog_mask` (or `leekrl_action_mask` for any
list of actions) checks the current state, inventory,
resources, cooldowns, range, line of sight and applicable generator rules.
Recompute after every action. `leekrl_action_execute` rechecks and executes
the selected action; return from the generator's AI callback to end a turn.
A valid cast need not hit an entity or have a useful effect:
`leekrl_action_catalog_trimmed_mask` leaves out useless casts, and those that can only
harm their own team or help the enemy (see `adr/observation.md`). Summons require
registered templates. They get an AI, so they play turns of their own.

`leekrl_action_catalog_create` builds a fixed catalog from **all currently
registered** weapons and chips: end turn, 840 movement destinations, then weapons
and chips ordered by item ID, with each item's offsets ordered by dy then dx.
Register the complete intended item set before creating catalogs. Catalogs
created from the same definitions have identical indices regardless of
equipment. Unequipped items are masked. Adding or changing definitions changes
the catalog and can invalidate saved models. Chips that only awakening plants
own are left out, since plants are scripted.

Movement slots cover all offsets with Manhattan distance 1 through 20.
The mask requires a free destination and a traversable path no longer than
both the actor's current MP and 20 steps; execution follows a shortest path.
This is a **per-action limit**, not a cap on the actor's MP. With more than
20 MP, destinations requiring over 20 path steps cannot be selected in one
action, though multiple movement actions can still spend the remaining MP.
Changing this limit changes action indices and invalidates saved models.

The generator's C data loaders are stubs. When `weapons.json`, `chips.json`,
`summons.json` and `components.json` are present at the project root, CMake uses
Python 3 to validate them with the standard `json` module and generate a C snapshot in
the build directory. Python is only needed for generation, not at runtime.
The supplied files are a snapshot; their provenance and currentness are not
verified. Confirm redistribution rights before sharing them.

Call `leekrl_item_snapshot_register` from `leekrl/item_snapshot.h` before
creating battles or action catalogs. It requires empty weapon, chip **and**
summon registries; the generator cannot enumerate summons to verify this.
Destroy all battles first, then call `leekrl_item_snapshot_clear`. Do not
modify registries or run generator sessions concurrently. Registration errors
are reported through its error buffer. `LEEKRL_HAS_ITEM_SNAPSHOT` is defined
when the files were available at CMake configure time.

The snapshot hash returned by `leekrl_item_snapshot_hash` identifies the
exact input bytes. Keep it with trained models: changing the data may change
action indices. Weapon equipment IDs come from JSON `item`, not record `id`.
If any files are absent, the core library builds without a snapshot; add all
four files and reconfigure to enable it.

The current generator limits entity inventories to 32 chips, even though the
game's equipment model is not inherently limited to that number.

### Multi-action battle environment

`leekrl/battle.h` provides a separate 1v1 environment against an idle
opponent. Each step executes one legal movement or cast, or ends the turn, for
the learner's leek or one of its summons: observation and mask are those of the
entity whose turn it is. Movement and casts can be chained within one turn.
Create two `LwEntityInfo`
builds, initialize `leekrl_battle_options`, then create a battle. Create the
Gym wrapper with `leekrl_battle_observation_size`, the count from
`leekrl_battle_catalog`, and `leekrl_battle_callbacks`. See
`tests/test_battle.c` for an end-to-end example including PPO collection.
Different builds can use separate battle contexts with the same item
registries and therefore the same action indices.

Observations follow `specs/Observation.md` and are written by
`leekrl/observation.h`: floats, and one category byte per map cell. The generic
environment only handles floats, so the battle appends the map as floats. A
situation the layout cannot represent fails the step: see `leekrl_battle_error`.

Rewards follow `specs/Rewards.md` and are written by `leekrl/reward.h`; the parts of
the last one are available with `leekrl_battle_reward_parts`. Turn-limit
draws terminate; a configurable decision limit truncates with a valid final
observation and mask for bootstrapping. Seeds use their low 31 bits.

The fight stays suspended between steps, so a step only executes its own action
(`lw_fight_step_*` in the submodule's `lw_fight.h`). Registries and item definitions must
remain unchanged while contexts are in use. For self-play and random builds, see
the batched environment below.

`tests/test_stepping.c` plays random fights through the engine's AI callback,
step by step, and through clones, and requires identical results. Pass a fight
count and a first seed to run more: `test_stepping 5000 1`.

### Batched self-play environment

`leekrl/vec.h` plays many 1v1 fights side by side with several threads, the agent
controlling both teams. Finished fights are replaced at once, with random builds or,
for 1% of the sides by default, a known build. Results depend on the seed and the
actions, never on the number of threads.

`python/leekrl` wraps it, with NumPy arrays written directly by C:

```python
from leekrl import VecEnv, load_known_builds

env = VecEnv(256, seed=1, known_builds=load_known_builds())
env.reset()
env.step(actions)  # one legal action per environment
```

Rewards belong to the previous decision of the team that decides now: see `vec.h`.
The package looks for the C library under `build*/`, or at `LEEKRL_LIBRARY`. Known
builds are read from `builds/known_builds.json`; `known_builds.example.json` shows the
format, where characteristics include components.

### Training

PyTorch, NumPy, and W&B if used, must be installed. From the project root, after building:

```sh
set PYTHONPATH=python
python -m leekrl.train --name first-run --updates 1000
```

Every setting of `Config` in `python/leekrl/train.py` is an option. Runs go to
`runs/<name>`: settings, `metrics.jsonl` and checkpoints. `--deterministic` gives identical
results for identical settings on the same machine. `--wandb online` sends metrics and
settings to Weights & Biases, after `wandb login`, without the email, computer name or
paths W&B normally adds; `--wandb offline` keeps them in the run folder. To stop early, press Ctrl+C or create a file named `stop` in the run's folder: the
current update is finished and a checkpoint saved, to continue with `--resume`.

The network's size is set by `--branch` (width of what each kind of input becomes),
`--hidden` and `--layers`.

`python -m leekrl.peek --checkpoint runs/first-run/checkpoint_000050.pt` writes a local
HTML page to step through fights: what the network receives, decoded, and what it makes of
it. `--peek-every 50` writes one regularly during training.

`python -m leekrl.show_builds` writes another one about the random builds of training fights.

### Builds

`leekrl/build.h` draws random legal level 301 builds, reproducible from a 64-bit
state, validates builds and converts them to generator entities. Capital costs
mirror the game client's `costs.ts`. `leekrl_build_options` sets how specialized
builds are and which characteristics receive capital.

`leekrl/rules.h` lists the items excluded from 1v1 fights.

### Plants

Awakening plants (corn, chilli pepper) act in the middle of other entities'
actions. They are not controlled by the agent: `leekrl/plant_ai.h` documents
the deterministic script they follow, for both sides. From the agent's point of
view, what a plant does is part of the action that awakened it.

## Masked PPO

`leekrl/ppo.h` provides a seeded, one-hidden-layer policy/value network, masked
categorical sampling, trajectory storage, generalized advantage estimation
(GAE), and clipped PPO updates. The implementation uses full-batch gradient
descent; it is not a drop-in replacement for Stable-Baselines3 MaskablePPO.
The value head can evaluate a final observation without an action mask.

`leekrl/collect.h` connects the environment to the learner. Reset the
environment, create a collector and trajectory with matching dimensions, then
collect transitions, compute GAE, update PPO and clear the trajectory for the
next rollout. The collector converts float observations to doubles, preserves
the mask used for each action and resets completed episodes with successive
seeds. It distinguishes termination from truncation when bootstrapping. See
`tests/test_collect.c` for a complete training loop and error checks.
