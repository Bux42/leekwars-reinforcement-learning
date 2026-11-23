#ifndef LEEKRL_BATTLE_H
#define LEEKRL_BATTLE_H

#include "leekrl/actions.h"
#include "leekrl/env.h"
#include "lw_entity_info.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct leekrl_battle leekrl_battle;

typedef struct {
    int max_turns;
    size_t max_decisions;
    int grid_radius;
    float win_reward;
    float loss_reward;
    float draw_reward;
    float step_reward;
} leekrl_battle_options;

enum {
    LEEKRL_BATTLE_TURN,
    LEEKRL_BATTLE_SELF_LIFE,
    LEEKRL_BATTLE_ENEMY_LIFE,
    LEEKRL_BATTLE_TP,
    LEEKRL_BATTLE_MP,
    LEEKRL_BATTLE_GRID
};

/*
 * Initialize defaults: 64 turns, 4096 decisions, grid radius 17,
 * terminal rewards +1/-1/0, and zero reward per step.
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
 * rather than ends the fight. Gym wrapper owns neither the context nor its
 * catalog. Destroy wrapper before context.
 */
leekrl_env_status leekrl_battle_create(const LwEntityInfo *self,
                                       const LwEntityInfo *opponent,
                                       const leekrl_battle_options *options,
                                       leekrl_battle **out);
void leekrl_battle_destroy(leekrl_battle *battle);
leekrl_env_callbacks leekrl_battle_callbacks(void);
size_t leekrl_battle_observation_size(const leekrl_battle *battle);
const leekrl_action_catalog *leekrl_battle_catalog(const leekrl_battle *battle);

#ifdef __cplusplus
}
#endif
#endif
