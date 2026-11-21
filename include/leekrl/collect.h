#ifndef LEEKRL_COLLECT_H
#define LEEKRL_COLLECT_H

#include "leekrl/env.h"
#include "leekrl/ppo.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct leekrl_collector leekrl_collector;

typedef enum {
    LEEKRL_COLLECT_OK = 0,
    LEEKRL_COLLECT_INVALID_ARGUMENT,
    LEEKRL_COLLECT_INVALID_STATE,
    LEEKRL_COLLECT_OUT_OF_MEMORY,
    LEEKRL_COLLECT_ENV_ERROR,
    LEEKRL_COLLECT_PPO_ERROR
} leekrl_collect_status;

typedef struct {
    size_t steps_collected;
    leekrl_env_status env_status;
    lr_status ppo_status;
} leekrl_collect_result;

/*
 * The caller owns env and model and must keep both alive. Reset env before the
 * first collection. action_seed controls policy sampling; next_episode_seed is
 * used for the first automatic reset, then incremented for each later reset.
 */
leekrl_collect_status leekrl_collector_create(leekrl_env *env, lr_ppo *model,
                                               uint64_t action_seed,
                                               uint64_t next_episode_seed,
                                               leekrl_collector **out_collector);
void leekrl_collector_destroy(leekrl_collector *collector);

/*
 * Append exactly steps transitions, automatically resetting after each episode.
 * The trajectory must have enough free capacity and matching dimensions.
 * On failure, result reports the number already stored and the underlying
 * error. After an environment error or a failure after stepping, reset the
 * environment and create a new collector before collecting again.
 */
leekrl_collect_status leekrl_collector_collect(leekrl_collector *collector,
                                                lr_trajectory *trajectory,
                                                size_t steps,
                                                leekrl_collect_result *result);

#ifdef __cplusplus
}
#endif
#endif
