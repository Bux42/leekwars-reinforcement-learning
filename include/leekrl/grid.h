#ifndef LEEKRL_GRID_H
#define LEEKRL_GRID_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct LwFight;
struct LwEntity;

#define LEEKRL_GRID_CHANNEL_COUNT 6

typedef enum {
    LEEKRL_GRID_BOUNDARY = 0,
    LEEKRL_GRID_OBSTACLE,
    LEEKRL_GRID_SELF,
    LEEKRL_GRID_ALLY,
    LEEKRL_GRID_ENEMY,
    LEEKRL_GRID_DEAD
} leekrl_grid_channel;

typedef enum {
    LEEKRL_GRID_OK = 0,
    LEEKRL_GRID_INVALID_ARGUMENT,
    LEEKRL_GRID_INVALID_STATE,
    LEEKRL_GRID_BUFFER_TOO_SMALL
} leekrl_grid_status;

/*
 * Channel-major, row-major observation:
 *   channel 0 boundary: map coordinate is outside the active map
 *   channel 1 obstacle:  cell is not walkable or contains an obstacle
 *   channel 2 self:      active entity occupies the cell
 *   channel 3 ally:      another living same-team entity occupies the cell
 *   channel 4 enemy:     living opposing-team entity occupies the cell
 *   channel 5 dead:      dead entity occupies the cell
 *
 * Rows and columns are ordered by dy from -radius to +radius and dx from
 * -radius to +radius. The observation is therefore relative to the active
 * entity's current (x,y), in generator map coordinates.
 */
typedef struct {
    int radius;
    size_t height;
    size_t width;
    size_t channels;
} leekrl_grid_layout;

/* Build a layout. Its flat float count is height * width * channels. */
leekrl_grid_status leekrl_grid_layout_for(int radius,
                                          leekrl_grid_layout *layout);

/*
 * Extract an agent-centered grid. fight and actor must belong to the same
 * live fight. Pass out=NULL and capacity=0 to query required. On
 * BUFFER_TOO_SMALL, required receives the needed count and out is untouched.
 */
leekrl_grid_status leekrl_grid_observation(struct LwFight *fight,
                                           struct LwEntity *actor,
                                           int radius,
                                           float *out,
                                           size_t capacity,
                                           size_t *required);

/* Shorter public spelling retained for Gym callback integrations. */
leekrl_grid_status leekrl_grid_extract(struct LwFight *fight,
                                       struct LwEntity *actor,
                                       int radius,
                                       float *out,
                                       size_t capacity,
                                       size_t *required);

const char *leekrl_grid_status_string(leekrl_grid_status status);

#ifdef __cplusplus
}
#endif
#endif
