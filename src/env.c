#include "leekrl/env.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct leekrl_env {
    size_t observation_size;
    size_t action_count;
    leekrl_env_callbacks callbacks;
    void *context;
    float *observation;
    uint8_t *mask;
    int valid;
    int done;
};

static int valid_output(const leekrl_env *env, int require_action)
{
    size_t i;
    int any_action = 0;
    for (i = 0; i < env->observation_size; ++i) {
        if (!isfinite(env->observation[i])) return 0;
    }
    for (i = 0; i < env->action_count; ++i) {
        if (env->mask[i] > 1) return 0;
        any_action |= env->mask[i];
    }
    return !require_action || any_action;
}

static void prepare_output(leekrl_env *env)
{
    size_t i;
    for (i = 0; i < env->observation_size; ++i) env->observation[i] = NAN;
    memset(env->mask, 2, env->action_count * sizeof(*env->mask));
}

leekrl_env_status leekrl_env_create(size_t observation_size, size_t action_count,
                                    leekrl_env_callbacks callbacks, void *context,
                                    leekrl_env **out_env)
{
    leekrl_env *env;
    if (!out_env) return LEEKRL_ENV_INVALID_ARGUMENT;
    *out_env = NULL;
    if (!observation_size || !action_count || !callbacks.reset || !callbacks.step ||
        observation_size > SIZE_MAX / sizeof(float)) {
        return LEEKRL_ENV_INVALID_ARGUMENT;
    }
    env = calloc(1, sizeof(*env));
    if (!env) return LEEKRL_ENV_OUT_OF_MEMORY;
    env->observation = calloc(observation_size, sizeof(*env->observation));
    env->mask = calloc(action_count, sizeof(*env->mask));
    if (!env->observation || !env->mask) {
        leekrl_env_destroy(env);
        return LEEKRL_ENV_OUT_OF_MEMORY;
    }
    env->observation_size = observation_size;
    env->action_count = action_count;
    env->callbacks = callbacks;
    env->context = context;
    *out_env = env;
    return LEEKRL_ENV_OK;
}

void leekrl_env_destroy(leekrl_env *env)
{
    if (!env) return;
    free(env->observation);
    free(env->mask);
    free(env);
}

leekrl_env_status leekrl_env_reset(leekrl_env *env, uint64_t seed)
{
    if (!env) return LEEKRL_ENV_INVALID_ARGUMENT;
    env->valid = 0;
    env->done = 0;
    prepare_output(env);
    if (env->callbacks.reset(env->context, seed, env->observation, env->mask)) {
        return LEEKRL_ENV_CALLBACK_FAILED;
    }
    if (!valid_output(env, 1)) return LEEKRL_ENV_INVALID_OUTPUT;
    env->valid = 1;
    return LEEKRL_ENV_OK;
}

leekrl_env_status leekrl_env_step(leekrl_env *env, size_t action,
                                  leekrl_step_result *result)
{
    float reward = NAN;
    int terminated = -1, truncated = -1;
    if (!env || !result) return LEEKRL_ENV_INVALID_ARGUMENT;
    if (!env->valid || env->done) return LEEKRL_ENV_INVALID_STATE;
    if (action >= env->action_count || !env->mask[action]) return LEEKRL_ENV_INVALID_ACTION;
    env->valid = 0;
    prepare_output(env);
    if (env->callbacks.step(env->context, action, env->observation, env->mask,
                            &reward, &terminated, &truncated)) {
        return LEEKRL_ENV_CALLBACK_FAILED;
    }
    if (!isfinite(reward) || (terminated != 0 && terminated != 1) ||
        (truncated != 0 && truncated != 1) ||
        !valid_output(env, !terminated && !truncated)) {
        return LEEKRL_ENV_INVALID_OUTPUT;
    }
    env->valid = 1;
    env->done = terminated || truncated;
    result->reward = reward;
    result->terminated = terminated;
    result->truncated = truncated;
    return LEEKRL_ENV_OK;
}

const float *leekrl_env_observation(const leekrl_env *env)
{
    return env && env->valid ? env->observation : NULL;
}

const uint8_t *leekrl_env_action_mask(const leekrl_env *env)
{
    return env && env->valid ? env->mask : NULL;
}

size_t leekrl_env_observation_size(const leekrl_env *env)
{
    return env ? env->observation_size : 0;
}

size_t leekrl_env_action_count(const leekrl_env *env)
{
    return env ? env->action_count : 0;
}

int leekrl_env_can_step(const leekrl_env *env)
{
    return env && env->valid && !env->done;
}

const char *leekrl_env_status_string(leekrl_env_status status)
{
    switch (status) {
    case LEEKRL_ENV_OK: return "ok";
    case LEEKRL_ENV_INVALID_ARGUMENT: return "invalid argument";
    case LEEKRL_ENV_OUT_OF_MEMORY: return "out of memory";
    case LEEKRL_ENV_INVALID_STATE: return "environment requires reset";
    case LEEKRL_ENV_INVALID_ACTION: return "invalid or masked action";
    case LEEKRL_ENV_CALLBACK_FAILED: return "environment callback failed";
    case LEEKRL_ENV_INVALID_OUTPUT: return "invalid environment output";
    }
    return "unknown environment status";
}
