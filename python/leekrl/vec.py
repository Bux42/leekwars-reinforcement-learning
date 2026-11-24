"""Many 1v1 self-play fights side by side. See include/leekrl/vec.h."""

import ctypes
import os

import numpy as np

from .native import STATS, VecBuffers, VecOptions, library

# Same orders as reward.h and vec.h.
REWARD_PARTS = ("life", "total_life", "summons", "outcome", "duration", "actions")
EPISODE_FIELDS = ("turns", "winner", "decisions_0", "decisions_1", "actions_0", "actions_1",
                  "life_0", "life_1", "known_0", "known_1", "redrawn")

# Instead of an action: the environment waits, and only its done is cleared.
WAIT = -1


def action_table():
    """What each action is: array (actions, 4) of kind, item, dx, dy.

    Kinds are END_TURN, MOVE, WEAPON and CHIP, from 0 to 3. The item is its position
    among the weapons, or the chips, of the observation. dx and dy are relative.
    """
    sizes = [ctypes.c_size_t() for _ in range(3)]
    lib = library()
    if not lib.leekrl_vec_sizes(*map(ctypes.byref, sizes)):
        raise RuntimeError("cannot compute the sizes of the environment")
    table = np.zeros((sizes[0].value, 4), np.int32)
    if not lib.leekrl_vec_action_table(table.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32))):
        raise RuntimeError("cannot describe the actions")
    return table


class VecEnv:
    """Outputs are arrays overwritten by each step, without any copy:

    floats       (n, floats)  float32
    map          (n, 2381)    uint8, cell categories
    mask         (n, bytes)   uint8, one bit per action, least significant first
    rewards      (n, 6)       float32, parts of the previous decision's reward
    team         (n,)         uint8, team that decides
    first        (n,)         uint8, first decision of that team in this fight
    done         (n,)         uint8, a fight ended and was replaced
    end_rewards  (n, 2, 6)    float32, per team, where done
    episode      (n, 11)      float32, see EPISODE_FIELDS, where done
    """

    def __init__(self, envs, threads=None, seed=0, known_builds=(), known_probability=None,
                 skip_forced=None, trim_useless=None, skew=None, stat_weights=None,
                 rewards=None):
        self._lib = library()
        self._vec = None
        layout = self._lib.layout
        if (layout.reward_parts, layout.episode_size) != (len(REWARD_PARTS),
                                                          len(EPISODE_FIELDS)):
            raise RuntimeError("the C library does not match this Python code: rebuild it")
        sizes = [ctypes.c_size_t() for _ in range(3)]
        if not self._lib.leekrl_vec_sizes(*map(ctypes.byref, sizes)):
            raise RuntimeError("cannot compute the sizes of the environment")
        self.actions, self.float_count, self.mask_bytes = (size.value for size in sizes)
        self.map_cells = layout.map_cells
        self.envs = envs

        options = VecOptions()
        self._lib.leekrl_vec_options_init(ctypes.byref(options))
        options.envs = envs
        options.threads = threads or max(1, (os.cpu_count() or 1) // 2)
        options.seed = seed
        if known_probability is not None:
            options.known_probability = known_probability
        if skip_forced is not None:
            options.skip_forced = skip_forced
        if trim_useless is not None:
            options.trim_useless = trim_useless
        if skew is not None:
            options.builds.skew = skew
        for stat, weight in (stat_weights or {}).items():
            options.builds.weights[STATS.index(stat)] = weight
        for name, value in (rewards or {}).items():
            if not hasattr(options.rewards, name):
                raise ValueError(f"unknown reward setting {name!r}")
            setattr(options.rewards, name, value)
        self.threads = options.threads

        parts = layout.reward_parts
        self.floats = np.zeros((envs, self.float_count), np.float32)
        self.map = np.zeros((envs, self.map_cells), np.uint8)
        self.mask = np.zeros((envs, self.mask_bytes), np.uint8)
        self.rewards = np.zeros((envs, parts), np.float32)
        self.team = np.zeros(envs, np.uint8)
        self.first = np.zeros(envs, np.uint8)
        self.done = np.zeros(envs, np.uint8)
        self.end_rewards = np.zeros((envs, 2, parts), np.float32)
        self.episode = np.zeros((envs, layout.episode_size), np.float32)
        buffers = VecBuffers()
        for name, kind in VecBuffers._fields_:
            setattr(buffers, name, getattr(self, name).ctypes.data_as(kind))

        error = ctypes.create_string_buffer(256)
        self._vec = self._lib.leekrl_vec_create(ctypes.byref(options), ctypes.byref(buffers),
                                                error, len(error))
        if not self._vec:
            raise ValueError(error.value.decode())
        self.known_names = []
        for name, weight, build in known_builds:
            if not self._lib.leekrl_vec_add_build(self._vec, ctypes.byref(build), weight,
                                                  error, len(error)):
                message = error.value.decode()
                self.close()
                raise ValueError(f"build {name!r}: {message}")
            self.known_names.append(name)

    def _check(self, ok):
        if not ok:
            raise RuntimeError(self._lib.leekrl_vec_error(self._vec).decode())

    def _open(self):
        if not self._vec:
            raise RuntimeError("the environment is closed")

    def reset(self):
        self._open()
        self._check(self._lib.leekrl_vec_reset(self._vec))

    def step(self, actions):
        """One legal action per environment, or WAIT. The GIL is released meanwhile."""
        self._open()
        actions = np.ascontiguousarray(actions, np.int32)
        if actions.shape != (self.envs,):
            raise ValueError(f"{self.envs} actions expected, not {actions.shape}")
        self._check(self._lib.leekrl_vec_step(
            self._vec, actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))))

    def legal(self):
        """The masks as booleans, (n, actions). A copy."""
        return np.unpackbits(self.mask, axis=1, count=self.actions,
                             bitorder="little").astype(bool)

    @property
    def data_hash(self):
        """Identifies the game data: keep it with trained models."""
        return self._lib.leekrl_item_snapshot_hash().decode()

    def close(self):
        if self._vec:
            self._lib.leekrl_vec_destroy(self._vec)
            self._vec = None

    def __del__(self):
        if getattr(self, "_vec", None):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
