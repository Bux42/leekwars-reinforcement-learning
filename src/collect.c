#include "leekrl/collect.h"

#include <stdlib.h>
#include <string.h>

struct leekrl_collector {
    leekrl_env *env;
    lr_ppo *model;
    lr_rng rng;
    uint64_t next_seed;
    double *observation;
    double *next_observation;
    unsigned char *mask;
    size_t observation_size;
    size_t action_count;
    int faulted;
};

leekrl_collect_status leekrl_collector_create(leekrl_env *env, lr_ppo *model,
                                               uint64_t action_seed,
                                               uint64_t next_episode_seed,
                                               leekrl_collector **out_collector)
{
    leekrl_collector *collector;
    size_t obs, actions;
    if (!out_collector) return LEEKRL_COLLECT_INVALID_ARGUMENT;
    *out_collector = NULL;
    if (!env || !model) return LEEKRL_COLLECT_INVALID_ARGUMENT;
    obs = leekrl_env_observation_size(env);
    actions = leekrl_env_action_count(env);
    if (!obs || !actions || obs > SIZE_MAX / sizeof(double) ||
        obs != lr_ppo_observation_size(model) ||
        actions != lr_ppo_action_count(model)) return LEEKRL_COLLECT_INVALID_ARGUMENT;
    collector = calloc(1, sizeof(*collector));
    if (!collector) return LEEKRL_COLLECT_OUT_OF_MEMORY;
    collector->observation = malloc(obs * sizeof(*collector->observation));
    collector->next_observation = malloc(obs * sizeof(*collector->next_observation));
    collector->mask = malloc(actions * sizeof(*collector->mask));
    if (!collector->observation || !collector->next_observation || !collector->mask) {
        leekrl_collector_destroy(collector);
        return LEEKRL_COLLECT_OUT_OF_MEMORY;
    }
    collector->env = env;
    collector->model = model;
    collector->observation_size = obs;
    collector->action_count = actions;
    collector->next_seed = next_episode_seed;
    lr_rng_seed(&collector->rng, action_seed);
    *out_collector = collector;
    return LEEKRL_COLLECT_OK;
}

void leekrl_collector_destroy(leekrl_collector *collector)
{
    if (!collector) return;
    free(collector->observation);
    free(collector->next_observation);
    free(collector->mask);
    free(collector);
}

leekrl_collect_status leekrl_collector_collect(leekrl_collector *collector,
                                                lr_trajectory *trajectory,
                                                size_t steps,
                                                leekrl_collect_result *result)
{
    size_t k, i, capacity, used;
    if (result) {
        result->steps_collected = 0;
        result->env_status = LEEKRL_ENV_OK;
        result->ppo_status = LR_OK;
    }
    if (!collector || !trajectory || !result || !steps) return LEEKRL_COLLECT_INVALID_ARGUMENT;
    if (collector->faulted) return LEEKRL_COLLECT_INVALID_STATE;
    capacity = lr_trajectory_capacity(trajectory);
    used = lr_trajectory_size(trajectory);
    if (lr_trajectory_observation_size(trajectory) != collector->observation_size ||
        lr_trajectory_action_count(trajectory) != collector->action_count ||
        used > capacity || steps > capacity - used) return LEEKRL_COLLECT_INVALID_ARGUMENT;
    if (!leekrl_env_can_step(collector->env)) {
        result->env_status = LEEKRL_ENV_INVALID_STATE;
        collector->faulted = 1;
        return LEEKRL_COLLECT_ENV_ERROR;
    }
    for (k = 0; k < steps; ++k) {
        const float *obs = leekrl_env_observation(collector->env);
        const uint8_t *mask = leekrl_env_action_mask(collector->env);
        leekrl_step_result step;
        size_t action;
        double logp, value, next_value = 0.0;
        for (i = 0; i < collector->observation_size; ++i)
            collector->observation[i] = obs[i];
        memcpy(collector->mask, mask, collector->action_count);
        result->ppo_status = lr_ppo_sample(collector->model, collector->observation,
                                           collector->mask, &collector->rng, &action,
                                           &logp, &value);
        if (result->ppo_status != LR_OK) return LEEKRL_COLLECT_PPO_ERROR;
        result->env_status = leekrl_env_step(collector->env, action, &step);
        if (result->env_status != LEEKRL_ENV_OK) {
            collector->faulted = 1;
            return LEEKRL_COLLECT_ENV_ERROR;
        }
        if (!step.terminated) {
            obs = leekrl_env_observation(collector->env);
            for (i = 0; i < collector->observation_size; ++i)
                collector->next_observation[i] = obs[i];
            result->ppo_status = lr_ppo_value(collector->model,
                                              collector->next_observation, &next_value);
            if (result->ppo_status != LR_OK) {
                collector->faulted = 1;
                return LEEKRL_COLLECT_PPO_ERROR;
            }
        }
        result->ppo_status = lr_trajectory_push(trajectory, collector->observation,
                                                collector->mask, action, logp, value,
                                                step.reward, next_value,
                                                step.terminated, step.truncated);
        if (result->ppo_status != LR_OK) {
            collector->faulted = 1;
            return LEEKRL_COLLECT_PPO_ERROR;
        }
        ++result->steps_collected;
        if (step.terminated || step.truncated) {
            result->env_status = leekrl_env_reset(collector->env, collector->next_seed);
            if (result->env_status != LEEKRL_ENV_OK) {
                collector->faulted = 1;
                return LEEKRL_COLLECT_ENV_ERROR;
            }
            ++collector->next_seed;
        }
    }
    return LEEKRL_COLLECT_OK;
}
