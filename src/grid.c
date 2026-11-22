#include "leekrl/grid.h"

#include <limits.h>
#include <stddef.h>

#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_map.h"
#include "lw_state.h"

#define LEEKRL_GRID_MAX_RADIUS 32

leekrl_grid_status leekrl_grid_layout_for(int radius,
                                          leekrl_grid_layout *layout)
{
    size_t side;
    if (!layout || radius < 0 || radius > LEEKRL_GRID_MAX_RADIUS)
        return LEEKRL_GRID_INVALID_ARGUMENT;
    side = (size_t)(2 * radius + 1);
    layout->radius = radius;
    layout->height = side;
    layout->width = side;
    layout->channels = LEEKRL_GRID_CHANNEL_COUNT;
    return LEEKRL_GRID_OK;
}

leekrl_grid_status leekrl_grid_observation(struct LwFight *fight,
                                           struct LwEntity *actor,
                                           int radius,
                                           float *out,
                                           size_t capacity,
                                           size_t *required)
{
    leekrl_grid_layout layout;
    leekrl_grid_status status;
    LwState *state;
    LwMap *map;
    LwCell *origin;
    LwEntity **entities;
    int entity_count;
    int actor_team;
    size_t side, total, c, row, col;
    int ox, oy, dx, dy;

    if (!fight || !actor || !required)
        return LEEKRL_GRID_INVALID_ARGUMENT;
    status = leekrl_grid_layout_for(radius, &layout);
    if (status != LEEKRL_GRID_OK)
        return status;
    total = side = layout.height;
    if (side > SIZE_MAX / side ||
        side * side > SIZE_MAX / LEEKRL_GRID_CHANNEL_COUNT)
        return LEEKRL_GRID_INVALID_ARGUMENT;
    total = side * side * LEEKRL_GRID_CHANNEL_COUNT;
    *required = total;

    if (!out && !capacity) return LEEKRL_GRID_OK;
    if (!out || capacity < total)
        return LEEKRL_GRID_BUFFER_TOO_SMALL;

    state = lw_fight_get_state(fight);
    if (!state)
        return LEEKRL_GRID_INVALID_STATE;
    map = lw_state_get_map(state);
    origin = lw_entity_get_cell(actor);
    if (!map || !origin)
        return LEEKRL_GRID_INVALID_STATE;

    for (c = 0; c < total; ++c)
        out[c] = 0.0f;

    ox = lw_cell_get_x(origin);
    oy = lw_cell_get_y(origin);
    actor_team = lw_entity_get_team(actor);
    entities = lw_state_get_entities(state, &entity_count);
    if (!entities)
        entity_count = 0;

    for (row = 0; row < side; ++row) {
        dy = (int)row - radius;
        for (col = 0; col < side; ++col) {
            dx = (int)col - radius;
            size_t base = (row * side + col);
            LwCell *target = lw_map_get_cell_xy(map, ox + dx, oy + dy);
            size_t i;

            if (!target) {
                out[LEEKRL_GRID_BOUNDARY * side * side + base] = 1.0f;
                continue;
            }
            if (!lw_cell_is_walkable(target) ||
                lw_cell_get_obstacle(target) != 0)
                out[LEEKRL_GRID_OBSTACLE * side * side + base] = 1.0f;

            for (i = 0; i < (size_t)entity_count; ++i) {
                LwEntity *entity = entities[i];
                LwCell *cell;
                if (!entity)
                    continue;
                cell = lw_entity_get_cell(entity);
                if (!cell || lw_cell_get_x(cell) != lw_cell_get_x(target) ||
                    lw_cell_get_y(cell) != lw_cell_get_y(target))
                    continue;
                if (lw_entity_is_dead(entity)) {
                    out[LEEKRL_GRID_DEAD * side * side + base] = 1.0f;
                } else if (entity == actor) {
                    out[LEEKRL_GRID_SELF * side * side + base] = 1.0f;
                } else if (lw_entity_get_team(entity) == actor_team) {
                    out[LEEKRL_GRID_ALLY * side * side + base] = 1.0f;
                } else {
                    out[LEEKRL_GRID_ENEMY * side * side + base] = 1.0f;
                }
            }
        }
    }
    return LEEKRL_GRID_OK;
}

leekrl_grid_status leekrl_grid_extract(struct LwFight *fight,
                                       struct LwEntity *actor,
                                       int radius,
                                       float *out,
                                       size_t capacity,
                                       size_t *required)
{
    return leekrl_grid_observation(fight, actor, radius, out, capacity, required);
}

const char *leekrl_grid_status_string(leekrl_grid_status status)
{
    switch (status) {
    case LEEKRL_GRID_OK: return "ok";
    case LEEKRL_GRID_INVALID_ARGUMENT: return "invalid argument";
    case LEEKRL_GRID_INVALID_STATE: return "invalid state";
    case LEEKRL_GRID_BUFFER_TOO_SMALL: return "buffer too small";
    }
    return "unknown grid status";
}
