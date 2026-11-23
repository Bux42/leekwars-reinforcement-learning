#ifndef LEEKRL_ACTIONS_H
#define LEEKRL_ACTIONS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct LwEntity;
struct LwFight;
struct LwAttack;

#define LEEKRL_MAX_MOVE_DISTANCE 20
#define LEEKRL_MOVE_ACTION_COUNT \
    (2 * LEEKRL_MAX_MOVE_DISTANCE * (LEEKRL_MAX_MOVE_DISTANCE + 1))

/* AI identifier of every summon created by leekrl_action_execute. */
#define LEEKRL_SUMMON_AI_ID 1

typedef enum {
    LEEKRL_ACTION_END_TURN,
    LEEKRL_ACTION_MOVE,
    LEEKRL_ACTION_WEAPON,
    LEEKRL_ACTION_CHIP
} leekrl_action_kind;

/*
 * Each index in a caller-owned array is a fixed policy output. For movement,
 * dx/dy identify a destination at Manhattan distance 1..20. For weapons and chips,
 * item_id selects an equipped item and dx/dy is relative to the acting
 * entity's current cell in generator coordinates. End-turn ignores all fields.
 *
 * IDs, not inventory positions, identify items so random builds do not change
 * a slot's meaning. An unequipped item or target outside the map is masked.
 */
typedef struct {
    leekrl_action_kind kind;
    int item_id;
    int dx;
    int dy;
} leekrl_action;

typedef enum {
    LEEKRL_ACTION_OK = 0,
    LEEKRL_ACTION_INVALID_ARGUMENT,
    LEEKRL_ACTION_INVALID_STATE,
    LEEKRL_ACTION_ILLEGAL,
    LEEKRL_ACTION_ENGINE_ERROR,
    LEEKRL_ACTION_BUFFER_TOO_SMALL,
    LEEKRL_ACTION_OUT_OF_MEMORY
} leekrl_action_status;

typedef struct leekrl_action_catalog leekrl_action_catalog;

/*
 * Snapshot all currently registered weapons and chips into a stable layout:
 * end turn, 840 movement destinations, then weapons and chips sorted by ID
 * with their cast offsets in dy-then-dx order. Register the complete intended item
 * set, summon templates included, before creating the catalog; the generator
 * does not load it itself. Chips owned only by awakening plants are left out:
 * plants are scripted (see plant_ai.h).
 * The registry and item definitions must remain unchanged during its use.
 */
leekrl_action_status leekrl_action_catalog_create(leekrl_action_catalog **out);

/* Same, leaving out some items, for example those of rules.h. */
leekrl_action_status leekrl_action_catalog_create_without(const int *excluded,
                                                          size_t excluded_count,
                                                          leekrl_action_catalog **out);
void leekrl_action_catalog_destroy(leekrl_action_catalog *catalog);
size_t leekrl_action_catalog_count(const leekrl_action_catalog *catalog);
const leekrl_action *leekrl_action_catalog_actions(const leekrl_action_catalog *catalog);

/*
 * Enumerate all map-reachable relative offsets allowed by an attack's static
 * range and launch type, in dy-then-dx order. No current fight, equipment,
 * obstacles or line-of-sight are considered. Pass actions=NULL and capacity=0
 * to query the required count. On BUFFER_TOO_SMALL, required receives the
 * needed count and the output buffer is untouched.
 *
 * Use the same ordered item list and attack definitions across randomized
 * builds to preserve action indices. Build movement/end-turn slots separately.
 */
leekrl_action_status leekrl_action_cast_layout(const struct LwAttack *attack,
                                               leekrl_action_kind kind,
                                               int item_id,
                                               leekrl_action *actions,
                                               size_t capacity,
                                               size_t *required);

/*
 * Compute one byte (0 or 1) per action without executing it. fight and actor
 * must belong to the same live fight, during that actor's dispatch. Output is
 * valid only on success. A changed fight state requires recomputing the mask.
 *
 * Actions describe attempted casts: a legal target need not contain an entity
 * or cause damage. Summons use the generator's summon rules and require their
 * registered templates. Movement destinations require an unoccupied,
 * reachable cell, with a path of at most 20 steps and the actor's current MP.
 */
leekrl_action_status leekrl_action_mask(struct LwFight *fight,
                                        struct LwEntity *actor,
                                        const leekrl_action *actions,
                                        size_t count, uint8_t *mask);

/*
 * Revalidate and execute one action. engine_result receives the generator's
 * result (positive on success) or 0 for END_TURN. END_TURN only signals the
 * caller: end the turn to actually finish it. Summons get an AI and play turns
 * of their own. Awakening plants need a plant AI dispatch set on the fight,
 * else the engine reports a fatal error. On ENGINE_ERROR consult lw_state_get_fatal_error().
 */
leekrl_action_status leekrl_action_execute(struct LwFight *fight,
                                           struct LwEntity *actor,
                                           const leekrl_action *action,
                                           int *engine_result);

const char *leekrl_action_status_string(leekrl_action_status status);

#ifdef __cplusplus
}
#endif
#endif
