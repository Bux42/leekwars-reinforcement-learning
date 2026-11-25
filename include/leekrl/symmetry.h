#ifndef LEEKRL_SYMMETRY_H
#define LEEKRL_SYMMETRY_H

#include <stdlib.h>

/*
 * The 8 symmetries of the grid: quarter turns and reflections. The rules of the game
 * do not depend on them. 0 changes nothing. Bit 1 (2) negates dx, bit 0 (1) negates
 * dy, then bit 2 (4) swaps both.
 */
#define LEEKRL_SYMMETRIES 8

static inline void leekrl_turn(int symmetry, int *dx, int *dy)
{
    int x = symmetry & 2 ? -*dx : *dx, y = symmetry & 1 ? -*dy : *dy;
    *dx = symmetry & 4 ? y : x;
    *dy = symmetry & 4 ? x : y;
}

/* The one that brings (dx, dy) to 0 <= dy <= dx. */
static inline int leekrl_symmetry_facing(int dx, int dy)
{
    return (abs(dy) > abs(dx) ? 4 : 0) | (dx < 0 ? 2 : 0) | (dy < 0 ? 1 : 0);
}

#endif
