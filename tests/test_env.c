#include "leekrl/env.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>

typedef struct {
    int calls;
    int mode;
    uint64_t seed;
} test_context;

static int reset(void *ptr, uint64_t seed, float *observation, uint8_t *mask)
{
    test_context *ctx = ptr;
    ctx->seed = seed;
    ctx->calls = 0;
    observation[0] = (float)seed;
    mask[0] = 1;
    mask[1] = 0;
    if (ctx->mode == 3) mask[0] = 0;
    if (ctx->mode == 6) observation[0] = NAN;
    return ctx->mode == 4;
}

static int step(void *ptr, size_t action, float *observation, uint8_t *mask,
                float *reward, int *terminated, int *truncated)
{
    test_context *ctx = ptr;
    ++ctx->calls;
    observation[0] = 42.0f;
    if (ctx->mode != 9) *reward = (float)action + 1.0f;
    *terminated = ctx->mode == 0;
    *truncated = ctx->mode == 1;
    if (ctx->mode != 10) mask[0] = 0;
    mask[1] = (*terminated || *truncated) ? 0 : (ctx->mode == 2 ? 2 : 1);
    if (ctx->mode == 7) *reward = NAN;
    return ctx->mode == 8;
}

int main(void)
{
    test_context ctx = {0, 0, 0};
    leekrl_env_callbacks callbacks = {reset, step};
    leekrl_env_callbacks missing_step = {reset, NULL};
    leekrl_env *env = NULL;
    leekrl_step_result result;
    assert(leekrl_env_create(0, 2, callbacks, &ctx, &env) == LEEKRL_ENV_INVALID_ARGUMENT);
    assert(env == NULL);
    assert(leekrl_env_create(1, 0, callbacks, &ctx, &env) == LEEKRL_ENV_INVALID_ARGUMENT);
    assert(leekrl_env_create(SIZE_MAX, 2, callbacks, &ctx, &env) == LEEKRL_ENV_INVALID_ARGUMENT);
    assert(leekrl_env_create(1, 2, missing_step, &ctx, &env) == LEEKRL_ENV_INVALID_ARGUMENT);
    assert(leekrl_env_create(1, 2, callbacks, &ctx, NULL) == LEEKRL_ENV_INVALID_ARGUMENT);
    assert(leekrl_env_create(1, 2, callbacks, &ctx, &env) == LEEKRL_ENV_OK);
    assert(leekrl_env_observation_size(env) == 1);
    assert(leekrl_env_action_count(env) == 2);
    assert(!leekrl_env_can_step(env));
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_INVALID_STATE);
    assert(leekrl_env_reset(env, 7) == LEEKRL_ENV_OK);
    assert(leekrl_env_can_step(env));
    assert(ctx.seed == 7 && leekrl_env_observation(env)[0] == 7.0f);
    assert(leekrl_env_action_mask(env)[0] == 1);
    assert(leekrl_env_step(env, 1, &result) == LEEKRL_ENV_INVALID_ACTION);
    assert(leekrl_env_step(env, 2, &result) == LEEKRL_ENV_INVALID_ACTION);
    assert(ctx.calls == 0);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_OK);
    assert(result.reward == 1.0f && result.terminated && !result.truncated);
    assert(!leekrl_env_can_step(env));
    assert(leekrl_env_observation(env)[0] == 42.0f);
    assert(leekrl_env_action_mask(env)[0] == 0 && leekrl_env_action_mask(env)[1] == 0);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_INVALID_STATE);

    ctx.mode = 1;
    assert(leekrl_env_reset(env, 8) == LEEKRL_ENV_OK);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_OK);
    assert(!result.terminated && result.truncated);
    assert(leekrl_env_observation(env)[0] == 42.0f);

    ctx.mode = 2;
    assert(leekrl_env_reset(env, 9) == LEEKRL_ENV_OK);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_INVALID_OUTPUT);
    assert(leekrl_env_observation(env) == NULL);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_INVALID_STATE);
    ctx.mode = 3;
    assert(leekrl_env_reset(env, 0) == LEEKRL_ENV_INVALID_OUTPUT);
    ctx.mode = 4;
    assert(leekrl_env_reset(env, 0) == LEEKRL_ENV_CALLBACK_FAILED);
    ctx.mode = 5;
    assert(leekrl_env_reset(env, 1) == LEEKRL_ENV_OK);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_OK);
    assert(!result.terminated && !result.truncated);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_INVALID_ACTION);
    assert(leekrl_env_step(env, 1, &result) == LEEKRL_ENV_OK);
    ctx.mode = 6;
    assert(leekrl_env_reset(env, 1) == LEEKRL_ENV_INVALID_OUTPUT);
    assert(leekrl_env_action_mask(env) == NULL);
    ctx.mode = 7;
    assert(leekrl_env_reset(env, 1) == LEEKRL_ENV_OK);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_INVALID_OUTPUT);
    ctx.mode = 8;
    assert(leekrl_env_reset(env, 1) == LEEKRL_ENV_OK);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_CALLBACK_FAILED);
    assert(leekrl_env_observation(env) == NULL);
    ctx.mode = 9;
    assert(leekrl_env_reset(env, 1) == LEEKRL_ENV_OK);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_INVALID_OUTPUT);
    ctx.mode = 10;
    assert(leekrl_env_reset(env, 1) == LEEKRL_ENV_OK);
    assert(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_INVALID_OUTPUT);
    leekrl_env_destroy(env);
    return 0;
}
