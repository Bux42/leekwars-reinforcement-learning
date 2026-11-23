#ifndef LEEKRL_ITEM_SNAPSHOT_H
#define LEEKRL_ITEM_SNAPSHOT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Characteristics capital can be spent on, and components can change. */
typedef enum {
    LEEKRL_STAT_LIFE, LEEKRL_STAT_STRENGTH, LEEKRL_STAT_WISDOM, LEEKRL_STAT_AGILITY,
    LEEKRL_STAT_RESISTANCE, LEEKRL_STAT_SCIENCE, LEEKRL_STAT_MAGIC, LEEKRL_STAT_FREQUENCY,
    LEEKRL_STAT_CORES, LEEKRL_STAT_RAM, LEEKRL_STAT_TP, LEEKRL_STAT_MP, LEEKRL_STAT_COUNT
} leekrl_stat;

typedef struct {
    int id;
    const char *name;
    int stats[LEEKRL_STAT_COUNT];
} leekrl_component;

/* Components sorted by id. Available without registering the snapshot. */
const leekrl_component *leekrl_components(size_t *count);

/* Legendary weapons are called forgotten in the game data. */
int leekrl_weapon_forgotten(int item_id);

/*
 * Register the build-time item snapshot in empty generator registries.
 * The bulb registry must also be empty: the generator cannot enumerate it.
 * Do not mutate registries or run concurrent generator sessions while active.
 * Destroy all battles before clearing the snapshot.
 *
 * Returns 1 on success, 0 on failure; error receives a diagnostic when set.
 * Calling register again while the snapshot is active is a no-op.
 */
int leekrl_item_snapshot_register(char *error, size_t error_size);
void leekrl_item_snapshot_clear(void);

/* SHA-256 of the input file hashes: weapons, chips, summons, components. */
const char *leekrl_item_snapshot_hash(void);

#ifdef __cplusplus
}
#endif
#endif
