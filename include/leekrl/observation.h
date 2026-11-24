#ifndef LEEKRL_OBSERVATION_H
#define LEEKRL_OBSERVATION_H

#include "leekrl/actions.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 1v1 observation, as specified in specs/Observation.md: floats, and one
 * category byte per map cell, from the point of view of the acting entity.
 */
#define LEEKRL_OBS_MAP_RADIUS 34
#define LEEKRL_OBS_MAP_CELLS 2381
#define LEEKRL_OBS_MAP_CENTER 1190
#define LEEKRL_OBS_MAX_ENTITIES 18
#define LEEKRL_OBS_SUMMON_TYPES 13
#define LEEKRL_OBS_SUMMON_CHIPS 6
#define LEEKRL_OBS_STATES 5
#define LEEKRL_OBS_EFFECTS 31

enum {
    LEEKRL_CELL_OUTSIDE,
    LEEKRL_CELL_FREE,
    LEEKRL_CELL_OBSTACLE,
    LEEKRL_CELL_SELF,
    LEEKRL_CELL_ALLY_LEEK,
    LEEKRL_CELL_ALLY_SUMMON,
    LEEKRL_CELL_ENEMY_LEEK,
    LEEKRL_CELL_ENEMY_SUMMON
};

/* Global block, first. */
enum {
    LEEKRL_OBS_TURN,
    LEEKRL_OBS_ALLY_SUMMONS,
    LEEKRL_OBS_ENEMY_SUMMONS,
    LEEKRL_OBS_GLOBAL_SIZE
};

/* Offsets in an entity block. */
enum {
    LEEKRL_OBS_PRESENT = 0,
    LEEKRL_OBS_ALLY = 1,
    LEEKRL_OBS_TYPE = 2,
    LEEKRL_OBS_LIFE = LEEKRL_OBS_TYPE + 1 + LEEKRL_OBS_SUMMON_TYPES,
    LEEKRL_OBS_TOTAL_LIFE,
    LEEKRL_OBS_DX,
    LEEKRL_OBS_DY,
    LEEKRL_OBS_DISTANCE,
    LEEKRL_OBS_TP,
    LEEKRL_OBS_TOTAL_TP,
    LEEKRL_OBS_MP,
    LEEKRL_OBS_TOTAL_MP,
    LEEKRL_OBS_STRENGTH,
    LEEKRL_OBS_WISDOM,
    LEEKRL_OBS_AGILITY,
    LEEKRL_OBS_RESISTANCE,
    LEEKRL_OBS_SCIENCE,
    LEEKRL_OBS_MAGIC,
    LEEKRL_OBS_FREQUENCY,
    LEEKRL_OBS_POWER,
    LEEKRL_OBS_DAMAGE_RETURN,
    LEEKRL_OBS_ABSOLUTE_SHIELD,
    LEEKRL_OBS_RELATIVE_SHIELD,
    LEEKRL_OBS_STATE,  /* active, duration; for each state */
    LEEKRL_OBS_EFFECT = LEEKRL_OBS_STATE + 2 * LEEKRL_OBS_STATES, /* amount, duration */
    LEEKRL_OBS_ZONE = LEEKRL_OBS_EFFECT + 2 * LEEKRL_OBS_EFFECTS,
    LEEKRL_OBS_AWAKENED,
    LEEKRL_OBS_SUMMON_COOLDOWN,
    LEEKRL_OBS_ENTITY_SIZE = LEEKRL_OBS_SUMMON_COOLDOWN + LEEKRL_OBS_SUMMON_CHIPS
};

/* First entity block, and first float after the entity blocks: the items of the
 * acting team's leek, those of the enemy leek, then the uses left. */
#define LEEKRL_OBS_ENTITIES LEEKRL_OBS_GLOBAL_SIZE
#define LEEKRL_OBS_ITEMS \
    (LEEKRL_OBS_ENTITIES + LEEKRL_OBS_MAX_ENTITIES * LEEKRL_OBS_ENTITY_SIZE)

typedef struct leekrl_observer leekrl_observer;

/* The catalog gives the items and their order. It is not kept. Registered item
 * definitions must not change while the observer is used. NULL on failure. */
leekrl_observer *leekrl_observer_create(const leekrl_action_catalog *catalog);
void leekrl_observer_destroy(leekrl_observer *observer);

size_t leekrl_observer_float_count(const leekrl_observer *observer);

/* Sizes of the item blocks: per leek, equipped items (weapons then chips), chip
 * cooldowns and weapon in hand; then limited items. */
size_t leekrl_observer_weapons(const leekrl_observer *observer);
size_t leekrl_observer_chips(const leekrl_observer *observer);
size_t leekrl_observer_limited(const leekrl_observer *observer);

/* Position of an item in its list (weapons, chips or limited items), or -1. */
int leekrl_observer_weapon_index(const leekrl_observer *observer, int item_id);
int leekrl_observer_chip_index(const leekrl_observer *observer, int item_id);
int leekrl_observer_limited_index(const leekrl_observer *observer, int item_id);

/* Positions of effect and state ids in an entity block, or -1. */
int leekrl_obs_effect_index(int effect_id);
int leekrl_obs_state_index(int state_id);

/* Index of a relative cell in the map, or -1 if farther than the radius. */
int leekrl_obs_map_index(int dx, int dy);

/*
 * Write leekrl_observer_float_count floats and LEEKRL_OBS_MAP_CELLS bytes. actor
 * must be alive and in the turn order of the fight. Thread-safe if fights are
 * distinct. Returns 1, or 0 if the situation cannot be represented: the outputs
 * are then unspecified and error describes the problem. This is a limit of the
 * layout to fix, not something to ignore.
 */
int leekrl_observe(const leekrl_observer *observer, struct LwFight *fight,
                   struct LwEntity *actor, float *floats, uint8_t *map,
                   char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
