#include "leekrl/collect.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

typedef struct {
    size_t steps;
    size_t resets;
    uint64_t last_seed;
    int truncate;
    int fail;
} bandit;

static int bandit_reset(void *context, uint64_t seed, float *obs, uint8_t *mask)
{
    bandit *b = context;
    ++b->resets;
    b->last_seed = seed;
    obs[0] = 1.0f;
    mask[0] = 1;
    mask[1] = 1;
    mask[2] = 0;
    return 0;
}

static int bandit_step(void *context, size_t action, float *obs, uint8_t *mask,
                       float *reward, int *terminated, int *truncated)
{
    bandit *b = context;
    ++b->steps;
    if (b->fail) return 1;
    obs[0] = 2.0f;
    mask[0] = 0;
    mask[1] = 0;
    mask[2] = 0;
    *reward = b->truncate ? 1.0f : (action == 1 ? 1.0f : -1.0f);
    *terminated = !b->truncate;
    *truncated = b->truncate;
    return 0;
}

int main(void)
{
    bandit b = {0, 0, 0, 1, 0};
    leekrl_env_callbacks callbacks = {bandit_reset, bandit_step};
    leekrl_env *env = NULL;
    leekrl_collector *collector = NULL;
    leekrl_collect_result result;
    lr_trajectory *trajectory = NULL;
    lr_ppo *model = NULL;
    lr_ppo *wrong_model = NULL;
    lr_rng rng;
    double final_obs[] = {2.0}, initial_obs[] = {1.0};
    unsigned char mask[] = {1, 1, 0};
    double next_value, target, probs[3], before;
    lr_ppo_options options = {0.08, 0.2, 0.5, 0.01, 4};
    size_t i;

    CHECK(leekrl_env_create(1, 3, callbacks, &b, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 10) == LEEKRL_ENV_OK);
    lr_rng_seed(&rng, 47);
    CHECK(lr_ppo_create(&model, 1, 8, 3, &rng) == LR_OK);
    CHECK(lr_ppo_create(&wrong_model, 2, 8, 3, &rng) == LR_OK);
    CHECK(leekrl_collector_create(env, wrong_model, 48, 11, &collector) ==
          LEEKRL_COLLECT_INVALID_ARGUMENT);
    CHECK(collector == NULL);
    lr_ppo_destroy(wrong_model);
    CHECK(leekrl_collector_create(env, model, 48, 11, &collector) == LEEKRL_COLLECT_OK);
    CHECK(lr_trajectory_create(&trajectory, 64, 1, 3) == LR_OK);
    CHECK(leekrl_collector_collect(collector, trajectory, 65, &result) ==
          LEEKRL_COLLECT_INVALID_ARGUMENT);
    CHECK(b.steps == 0 && lr_trajectory_size(trajectory) == 0);

    CHECK(lr_ppo_value(model, final_obs, &next_value) == LR_OK);
    CHECK(leekrl_collector_collect(collector, trajectory, 2, &result) == LEEKRL_COLLECT_OK);
    CHECK(result.steps_collected == 2 && b.resets == 3 && b.last_seed == 12);
    CHECK(lr_trajectory_gae(trajectory, 0.9, 0.95) == LR_OK);
    CHECK(lr_trajectory_target(trajectory, 0, NULL, &target) == LR_OK);
    CHECK(fabs(target - (1.0 + 0.9 * next_value)) < 1e-12);
    CHECK(lr_trajectory_target(trajectory, 1, NULL, &target) == LR_OK);
    CHECK(fabs(target - (1.0 + 0.9 * next_value)) < 1e-12);

    b.truncate = 0;
    CHECK(lr_ppo_evaluate(model, initial_obs, mask, 1, probs, NULL, NULL, NULL) == LR_OK);
    before = probs[1];
    for (i = 0; i < 100; ++i) {
        lr_trajectory_clear(trajectory);
        CHECK(leekrl_collector_collect(collector, trajectory, 64, &result) == LEEKRL_COLLECT_OK);
        CHECK(result.steps_collected == 64);
        CHECK(lr_trajectory_gae(trajectory, 0.99, 0.95) == LR_OK);
        CHECK(lr_ppo_update(model, trajectory, &options) == LR_OK);
    }
    CHECK(lr_ppo_evaluate(model, initial_obs, mask, 1, probs, NULL, NULL, NULL) == LR_OK);
    CHECK(probs[1] > 0.95 && probs[1] > before + 0.1 && probs[2] == 0.0);
    lr_trajectory_clear(trajectory);
    b.fail = 1;
    CHECK(leekrl_collector_collect(collector, trajectory, 1, &result) ==
          LEEKRL_COLLECT_ENV_ERROR);
    CHECK(result.steps_collected == 0 && result.env_status == LEEKRL_ENV_CALLBACK_FAILED);
    CHECK(leekrl_collector_collect(collector, trajectory, 1, &result) ==
          LEEKRL_COLLECT_INVALID_STATE);
    lr_trajectory_destroy(trajectory);
    leekrl_collector_destroy(collector);
    lr_ppo_destroy(model);
    leekrl_env_destroy(env);
    puts("Rollout collection tests passed");
    return 0;
}
