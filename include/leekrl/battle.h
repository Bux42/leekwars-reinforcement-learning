#ifndef LEEKRL_BATTLE_H
#define LEEKRL_BATTLE_H

#include "leekrl/actions.h"
#include "leekrl/env.h"
#include "leekrl/observation.h"
#include "leekrl/reward.h"
#include "lw_entity_info.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct leekrl_battle leekrl_battle;

typedef struct {
    int max_turns;
    size_t max_decisions;
    leekrl_reward_options rewards;
} leekrl_battle_options;

/*
 * Initialize defaults: 64 turns, 4096 decisions, rewards of reward.h.
 */
void leekrl_battle_options_init(leekrl_battle_options *options);

/*
 * A 1v1 environment against an idle opponent. Copies both entity definitions,
 * including start effects. Register the complete intended weapon/chip catalog
 * and summon templates first; keep registries and their definitions unchanged
 * until destruction.
 *
 * Items of rules.h are not actions, and builds equipping one are rejected.
 *
 * A decision is one movement, cast, or end turn of the learner's leek or of one
 * of its summons: observation and mask are those of the entity whose turn it
 * is. Awakening plants are scripted (plant_ai.h), for both sides. The fight
 * stays suspended between steps, so a step only executes its own action. The
 * fixed catalog is independent of equipped items. The decision limit truncates
 * rather than ends the fight.
 *
 * The observation is that of observation.h: the floats, then the map categories
 * as floats. It is all zeros once the fight is over. Gym wrapper owns neither the context nor its
 * catalog. Destroy wrapper before context.
 */
leekrl_env_status leekrl_battle_create(const LwEntityInfo *self,
                                       const LwEntityInfo *opponent,
                                       const leekrl_battle_options *options,
                                       leekrl_battle **out);
void leekrl_battle_destroy(leekrl_battle *battle);
leekrl_env_callbacks leekrl_battle_callbacks(void);
size_t leekrl_battle_observation_size(const leekrl_battle *battle);
const leekrl_observer *leekrl_battle_observer(const leekrl_battle *battle);

/* Parts of the last reward, LEEKRL_REWARD_PARTS floats. */
const float *leekrl_battle_reward_parts(const leekrl_battle *battle);

/* Why the last reset or step failed, if the situation could not be observed. */
const char *leekrl_battle_error(const leekrl_battle *battle);

/* Life of the leek of team 0 (learner) or 1 (opponent), also after the fight. */
int leekrl_battle_life(const leekrl_battle *battle, int team);
const leekrl_action_catalog *leekrl_battle_catalog(const leekrl_battle *battle);

#ifdef __cplusplus
}
#endif
#endif
