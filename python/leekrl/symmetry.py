"""The 8 symmetries of the grid, as in include/leekrl/symmetry.h.

Training does not need them, the environment turns the world itself: they are for looking
at it, and for checking.
"""

import numpy as np

from .vec import action_table

RADIUS = 34
MAP_CELLS = 2381


def turn(symmetry, dx, dy):
    """Arrays or numbers. The symmetries can be an array too."""
    x = np.where(np.bitwise_and(symmetry, 2) > 0, -dx, dx)
    y = np.where(np.bitwise_and(symmetry, 1) > 0, -dy, dy)
    swap = np.bitwise_and(symmetry, 4) > 0
    return np.where(swap, y, x), np.where(swap, x, y)


def turn_back(symmetry, dx, dy):
    swap = np.bitwise_and(symmetry, 4) > 0
    x, y = np.where(swap, dy, dx), np.where(swap, dx, dy)
    return (np.where(np.bitwise_and(symmetry, 2) > 0, -x, x),
            np.where(np.bitwise_and(symmetry, 1) > 0, -y, y))


def facing(dx, dy):
    """The symmetry that brings (dx, dy) to 0 <= dy <= dx."""
    return (4 * (np.abs(dy) > np.abs(dx)) + 2 * (np.less(dx, 0)) + np.less(dy, 0)).astype(int)


def map_index(dx, dy):
    """Position of relative cells in the map of an observation."""
    above = (dy + RADIUS) ** 2
    below = MAP_CELLS - (RADIUS - dy + 1) ** 2
    return np.where(dy <= 0, above, below) + dx + RADIUS - np.abs(dy)


def turned_cells():
    """Where each cell of the map goes, (8, 2381)."""
    rows = [(x, y) for y in range(-RADIUS, RADIUS + 1)
            for x in range(abs(y) - RADIUS, RADIUS - abs(y) + 1)]
    x, y = np.array(rows).T
    return np.stack([map_index(*turn(s, x, y)) for s in range(8)])


def turned_actions():
    """What each action becomes, (8, actions)."""
    table = action_table()
    index = {tuple(row): i for i, row in enumerate(table.tolist())}
    result = np.zeros((8, len(table)), np.int64)
    for s in range(8):
        dx, dy = turn(s, table[:, 2], table[:, 3])
        rows = np.stack([table[:, 0], table[:, 1], dx, dy], 1).tolist()
        result[s] = [index[tuple(row)] for row in rows]
    return result
