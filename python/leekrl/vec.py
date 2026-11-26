"""Many 1v1 self-play fights side by side. See include/leekrl/vec.h."""

import ctypes
import json
import os

import numpy as np

from .native import STATS, VecBuffers, VecOptions, library

# Same orders as reward.h and vec.h.
REWARD_PARTS = ("life", "total_life", "summons", "outcome", "duration", "actions")
EPISODE_FIELDS = ("turns", "winner", "decisions_0", "decisions_1", "actions_0", "actions_1",
                  "life_0", "life_1", "known_0", "known_1", "redrawn")

# Casts left out of the masks, see actions.h. Each level also leaves out those of the
# previous one.
TRIMS = ("nothing", "useless", "harmful")
# Instead of an action: the environment waits, and only its done is cleared.
WAIT = -1


def sizes():
    """Numbers of actions, and of floats of an observation."""
    result = [ctypes.c_size_t() for _ in range(2)]
    if not library().leekrl_vec_sizes(*map(ctypes.byref, result)):
        raise RuntimeError("cannot compute the sizes of the environment")
    return tuple(size.value for size in result)


def action_table():
    """What each action is: array (actions, 4) of kind, item, dx, dy.

    Kinds are END_TURN, MOVE, WEAPON and CHIP, from 0 to 3. The item is its position
    among the weapons, or the chips, of the observation. dx and dy are relative.
    """
    table = np.zeros((sizes()[0], 4), np.int32)
    if not library().leekrl_vec_action_table(table.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32))):
        raise RuntimeError("cannot describe the actions")
    return table


def take(values, counts, rows):
    """The lists of some rows, from the lists of all of them, one after the other."""
    counts = counts.astype(np.int64)
    ends = counts.cumsum()
    chosen = counts[rows]
    starts = (ends - counts)[rows] - (chosen.cumsum() - chosen)
    return values[np.repeat(starts, chosen) + np.arange(chosen.sum())], chosen


class VecEnv:
    """Outputs are arrays overwritten by each step, without any copy:

    floats        (n, floats)  float32
    legal_counts  (n,)         int32, legal actions of each environment
    held_counts   (n,)         int32, cells that are not free
    sight_counts  (n,)         int32, cells from which the enemy leek is seen, if sight
    place         (n,)         int16, where the map is
    symmetry      (n,)         uint8, how the world is turned
    map           (n, 2381)    uint8, cell categories, if dense_map
    rewards       (n, 6)       float32, parts of the previous decision's reward
    team          (n,)         uint8, team that decides
    first         (n,)         uint8, first decision of that team in this fight
    done          (n,)         uint8, a fight ended and was replaced
    end_rewards   (n, 2, 6)    float32, per team, where done
    episode       (n, 11)      float32, see EPISODE_FIELDS, where done

    legal_actions, held and sight hold the lists of all the environments, one after the
    other: actions in order, cell x 6 + category - 2, and cells (see vec.h).

    allocate makes the arrays: pinned memory is sent faster to a GPU.
    """

    def __init__(self, envs, threads=None, seed=0, known_builds=(), known_probability=None,
                 skip_forced=None, trim=None, face_enemy=None, repeat_moves=None, skew=None,
                 stat_weights=None, rewards=None, dense_map=False, sight=False,
                 allocate=np.zeros):
        self._lib = library()
        self._vec = None
        layout = self._lib.layout
        if (layout.reward_parts, layout.episode_size) != (len(REWARD_PARTS),
                                                          len(EPISODE_FIELDS)):
            raise RuntimeError("the C library does not match this Python code: rebuild it")
        self.actions, self.float_count = sizes()
        self.map_cells = layout.map_cells
        self.envs = envs

        options = VecOptions()
        self._lib.leekrl_vec_options_init(ctypes.byref(options))
        options.envs = envs
        options.threads = threads or max(1, (os.cpu_count() or 1) // 2)
        options.seed = seed
        if trim is not None:
            if trim not in TRIMS:
                raise ValueError(f"trim must be one of {', '.join(TRIMS)}, not {trim!r}")
            options.trim = TRIMS.index(trim)
        for name, value in (("known_probability", known_probability),
                            ("skip_forced", skip_forced), ("face_enemy", face_enemy),
                            ("repeat_moves", repeat_moves), ("sight", sight)):
            if value is not None:
                setattr(options, name, value)
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
        self.floats = allocate((envs, self.float_count), np.float32)
        self._legal = allocate(envs * self.actions, np.int16)
        self._held = allocate(envs * layout.max_held, np.int16)
        self.legal_counts = np.zeros(envs, np.int32)
        self.held_counts = np.zeros(envs, np.int32)
        self._sight = allocate(envs * layout.max_sight, np.int16) if sight else None
        self.sight_counts = np.zeros(envs, np.int32) if sight else None
        self.place = np.zeros(envs, np.int16)
        self.symmetry = np.zeros(envs, np.uint8)
        self.map = np.zeros((envs, self.map_cells), np.uint8) if dense_map else None
        self.rewards = np.zeros((envs, parts), np.float32)
        self.team = np.zeros(envs, np.uint8)
        self.first = np.zeros(envs, np.uint8)
        self.done = np.zeros(envs, np.uint8)
        self.end_rewards = np.zeros((envs, 2, parts), np.float32)
        self.episode = np.zeros((envs, layout.episode_size), np.float32)
        buffers = VecBuffers()
        for name, kind in VecBuffers._fields_:
            array = getattr(self, {"legal": "_legal", "held": "_held",
                                   "sight": "_sight"}.get(name, name))
            if array is not None:
                setattr(buffers, name, array.ctypes.data_as(kind))

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

    def record(self, rows, on=True):
        """Record the fights that start in some environments from now on, or stop."""
        self._open()
        for row in np.atleast_1d(rows):
            if not self._lib.leekrl_vec_record(self._vec, int(row), bool(on)):
                raise IndexError(f"no environment {row}")

    def recorded_actions(self, row):
        """Actions of the fight recorded so far, in an environment: the next ones come from
        its decision. -1 if not recorded."""
        self._open()
        return self._lib.leekrl_vec_recorded_actions(self._vec, int(row))

    def report(self, row):
        """Where done, the recorded fight that ended: the "data" of a Leek Wars fight
        report, as a dictionary. None for fights that are not recorded."""
        self._open()
        complete = ctypes.c_int()
        text = self._lib.leekrl_vec_report(self._vec, int(row), ctypes.byref(complete))
        if text is None:
            return None
        if not complete.value:
            raise RuntimeError(f"the record of environment {row} misses actions")
        return json.loads(text)

    @property
    def legal_actions(self):
        return self._legal[:self.legal_counts.sum()]

    @property
    def held(self):
        return self._held[:self.held_counts.sum()]

    @property
    def sight(self):
        """None without the sight option."""
        if self._sight is None:
            return None
        return self._sight[:self.sight_counts.sum()]

    def legal(self):
        """The legal actions as booleans, (n, actions). A copy."""
        result = np.zeros((self.envs, self.actions), bool)
        result[np.repeat(np.arange(self.envs), self.legal_counts), self.legal_actions] = True
        return result

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
