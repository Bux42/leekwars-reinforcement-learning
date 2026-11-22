#ifndef LEEKRL_ITEM_SNAPSHOT_H
#define LEEKRL_ITEM_SNAPSHOT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/* SHA-256 of the three input file hashes, in weapons/chips/summons order. */
const char *leekrl_item_snapshot_hash(void);

#ifdef __cplusplus
}
#endif
#endif
