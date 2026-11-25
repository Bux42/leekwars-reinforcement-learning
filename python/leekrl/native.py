"""Loading of the C library and declaration of its functions."""

import ctypes
import os
from ctypes import (POINTER, Structure, c_char_p, c_double, c_float, c_int, c_int16,
                    c_int32, c_size_t, c_uint8, c_uint64, c_void_p)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("leekrl.dll", "libleekrl.so", "leekrl.so", "libleekrl.dylib", "leekrl.dylib")

# Same order as leekrl_stat in item_snapshot.h.
STATS = ("life", "strength", "wisdom", "agility", "resistance", "science", "magic",
         "frequency", "cores", "ram", "tp", "mp")


class Build(Structure):
    _fields_ = [("components", c_int * 8), ("bought", c_int * len(STATS)),
                ("weapons", c_int * 4), ("chips", c_int * 32), ("n_chips", c_int)]


class BuildOptions(Structure):
    _fields_ = [("skew", c_double), ("weights", c_double * len(STATS))]


class RewardOptions(Structure):
    _fields_ = [(name, c_float) for name in (
        "win", "loss", "draw", "duration", "actions", "action_scale", "life", "total_life",
        "summon", "summon_life")]


class VecOptions(Structure):
    _fields_ = [("envs", c_int), ("threads", c_int), ("seed", c_uint64),
                ("known_probability", c_double), ("skip_forced", c_int),
                ("trim", c_int), ("face_enemy", c_int), ("repeat_moves", c_int),
                ("builds", BuildOptions), ("rewards", RewardOptions)]


class VecBuffers(Structure):
    _fields_ = [("floats", POINTER(c_float)), ("legal", POINTER(c_int16)),
                ("legal_counts", POINTER(c_int32)), ("held", POINTER(c_int16)),
                ("held_counts", POINTER(c_int32)), ("place", POINTER(c_int16)),
                ("symmetry", POINTER(c_uint8)), ("map", POINTER(c_uint8)),
                ("rewards", POINTER(c_float)), ("team", POINTER(c_uint8)),
                ("first", POINTER(c_uint8)), ("done", POINTER(c_uint8)),
                ("end_rewards", POINTER(c_float)), ("episode", POINTER(c_float))]


class Layout(Structure):
    """Sizes known by C, compared with what Python assumes."""
    _fields_ = [(name, c_int) for name in (
        "map_cells", "max_held", "reward_parts", "episode_size", "stats", "options_size", "buffers_size",
        "build_size")]


def find_library():
    """LEEKRL_LIBRARY, else the most recent library under build*/."""
    path = os.environ.get("LEEKRL_LIBRARY")
    if path:
        return Path(path)
    found = [file for folder in ROOT.glob("build*") for name in NAMES
             for file in folder.rglob(name)]
    if not found:
        raise FileNotFoundError(
            "C library not found: build the project, or set LEEKRL_LIBRARY")
    return max(found, key=lambda file: file.stat().st_mtime)


def _declare(library):
    size = POINTER(c_size_t)
    declarations = {
        "leekrl_vec_layout": (None, [POINTER(Layout)]),
        "leekrl_item_snapshot_register": (c_int, [c_char_p, c_size_t]),
        "leekrl_item_snapshot_hash": (c_char_p, []),
        "leekrl_vec_options_init": (None, [POINTER(VecOptions)]),
        "leekrl_vec_sizes": (c_int, [size, size]),
        "leekrl_vec_create": (c_void_p, [POINTER(VecOptions), POINTER(VecBuffers), c_char_p,
                                         c_size_t]),
        "leekrl_vec_destroy": (None, [c_void_p]),
        "leekrl_vec_add_build": (c_int, [c_void_p, POINTER(Build), c_double, c_char_p,
                                         c_size_t]),
        "leekrl_vec_reset": (c_int, [c_void_p]),
        "leekrl_vec_step": (c_int, [c_void_p, POINTER(c_int32)]),
        "leekrl_vec_error": (c_char_p, [c_void_p]),
        "leekrl_build_base": (c_int, [c_int]),
        "leekrl_build_stat": (c_int, [POINTER(Build), c_int]),
        "leekrl_build_options_init": (None, [POINTER(BuildOptions)]),
        "leekrl_build_random": (c_int, [c_void_p, POINTER(BuildOptions), POINTER(c_uint64),
                                        POINTER(Build)]),
        "leekrl_build_pool_create": (c_void_p, [c_void_p]),
        "leekrl_build_pool_destroy": (None, [c_void_p]),
        "leekrl_action_catalog_create_without": (c_int, [POINTER(c_int), c_size_t,
                                                         POINTER(c_void_p)]),
        "leekrl_action_catalog_destroy": (None, [c_void_p]),
        "leekrl_vec_action_table": (c_int, [POINTER(c_int32)]),
        "leekrl_vec_items": (c_int, [POINTER(c_int32), POINTER(c_int32)]),
    }
    for name, (result, arguments) in declarations.items():
        function = getattr(library, name)
        function.restype = result
        function.argtypes = arguments
    return library


_library = None


def library():
    """The library, loaded once, with the game data registered."""
    global _library
    if _library is None:
        path = find_library()
        loaded = _declare(ctypes.CDLL(str(path)))
        layout = Layout()
        loaded.leekrl_vec_layout(ctypes.byref(layout))
        expected = (len(STATS), ctypes.sizeof(VecOptions), ctypes.sizeof(VecBuffers),
                    ctypes.sizeof(Build))
        if (layout.stats, layout.options_size, layout.buffers_size,
                layout.build_size) != expected:
            raise RuntimeError(f"{path} does not match this Python code: rebuild it")
        error = ctypes.create_string_buffer(512)
        if not loaded.leekrl_item_snapshot_register(error, len(error)):
            raise RuntimeError(error.value.decode())
        loaded.layout = layout
        _library = loaded
    return _library
