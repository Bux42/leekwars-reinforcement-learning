#ifndef LEEKRL_RULES_H
#define LEEKRL_RULES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Item IDs a leek cannot equip in 1v1 fights: items of bosses and of their
 * fights, and items not supported yet. Weapons and chips share one ID space. Chips owned by plants are not
 * listed: the action catalog finds them in the summon templates.
 * Changing this list changes the 1v1 action indices.
 */
extern const int leekrl_solo_excluded_items[];
extern const size_t leekrl_solo_excluded_count;

int leekrl_solo_item_excluded(int item_id);

#ifdef __cplusplus
}
#endif
#endif
