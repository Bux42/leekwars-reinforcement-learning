#ifndef LEEKRL_ENV_H
#define LEEKRL_ENV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LEEKRL_ENV_OK = 0,
    LEEKRL_ENV_INVALID_ARGUMENT,
    LEEKRL_ENV_OUT_OF_MEMORY,
    LEEKRL_ENV_INVALID_STATE,
    LEEKRL_ENV_INVALID_ACTION,
    LEEKRL_ENV_CALLBACK_FAILED,
    LEEKRL_ENV_INVALID_OUTPUT
} leekrl_env_status;

typedef struct leekrl_env leekrl_env;

/*
 * Callbacks write observation_size finite floats and action_count mask bytes.
 * Mask bytes must be 0 or 1. At least one action must be legal after reset and
 * after non-final steps. Final observations remain available for bootstrapping
 * truncations. Return 0 on success; any other value reports a callback error.
 *
 * The caller owns context and must keep it alive until the environment is
 * destroyed. A failed callback or invalid output requires a successful reset.
 */
typedef struct {
    int (*reset)(void *context, uint64_t seed, float *observation,
                 uint8_t *action_mask);
    int (*step)(void *context, size_t action, float *observation,
                uint8_t *action_mask, float *reward, int *terminated,
                int *truncated);
} leekrl_env_callbacks;

typedef struct {
    float reward;
    int terminated;
    int truncated;
} leekrl_step_result;

leekrl_env_status leekrl_env_create(size_t observation_size, size_t action_count,
                                    leekrl_env_callbacks callbacks, void *context,
                                    leekrl_env **out_env);
void leekrl_env_destroy(leekrl_env *env);
leekrl_env_status leekrl_env_reset(leekrl_env *env, uint64_t seed);
leekrl_env_status leekrl_env_step(leekrl_env *env, size_t action,
                                  leekrl_step_result *result);

/* Returned buffers are owned by env and valid until its next reset/step/destroy.
 * They are NULL until reset succeeds or after an error invalidates the state. */
const float *leekrl_env_observation(const leekrl_env *env);
const uint8_t *leekrl_env_action_mask(const leekrl_env *env);
size_t leekrl_env_observation_size(const leekrl_env *env);
size_t leekrl_env_action_count(const leekrl_env *env);
int leekrl_env_can_step(const leekrl_env *env);
const char *leekrl_env_status_string(leekrl_env_status status);

#ifdef __cplusplus
}
#endif
#endif
