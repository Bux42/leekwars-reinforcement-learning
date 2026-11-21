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

The submodule tracks the `java-parity` branch for `git submodule update --remote`.
Normal checkouts use the commit recorded by this repository.

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
