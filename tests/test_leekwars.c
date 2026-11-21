#include "leekrl/leekwars.h"
#include "lw_constants.h"
#include "lw_effect_params.h"
#include "lw_weapon.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

static int exercise_masks(int weapon_id, int *approaches, int *fires)
{
    leekrl_leekwars *context;
    leekrl_env *env;
    leekrl_step_result result;
    size_t history[8], length, j, action, chosen;
    uint64_t seed;
    uint8_t mask[LEEKRL_LW_ACTION_COUNT];
    float observation[LEEKRL_LW_OBSERVATION_SIZE];

    CHECK(leekrl_leekwars_create(weapon_id, 8, &context) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_create(LEEKRL_LW_OBSERVATION_SIZE, LEEKRL_LW_ACTION_COUNT,
                            leekrl_leekwars_callbacks(), context, &env) == LEEKRL_ENV_OK);
    for (seed = 0; seed < 16; ++seed) {
        CHECK(leekrl_env_reset(env, seed) == LEEKRL_ENV_OK);
        for (length = 0; length < 8; ++length) {
            memcpy(mask, leekrl_env_action_mask(env), sizeof(mask));
            memcpy(observation, leekrl_env_observation(env), sizeof(observation));
            for (action = 0; action < LEEKRL_LW_ACTION_COUNT; ++action) {
                if (!mask[action]) continue;
                CHECK(leekrl_env_reset(env, seed) == LEEKRL_ENV_OK);
                for (j = 0; j < length; ++j) {
                    CHECK(leekrl_env_step(env, history[j], &result) == LEEKRL_ENV_OK);
                    CHECK(!result.terminated && !result.truncated);
                }
                CHECK(memcmp(mask, leekrl_env_action_mask(env), sizeof(mask)) == 0);
                CHECK(memcmp(observation, leekrl_env_observation(env),
                             sizeof(observation)) == 0);
                CHECK(leekrl_env_step(env, action, &result) == LEEKRL_ENV_OK);
                if (action == LEEKRL_LW_APPROACH) ++*approaches;
                if (action == LEEKRL_LW_FIRE) ++*fires;
            }
            chosen = mask[LEEKRL_LW_FIRE] ? LEEKRL_LW_FIRE :
                     mask[LEEKRL_LW_APPROACH] ? LEEKRL_LW_APPROACH : LEEKRL_LW_WAIT;
            CHECK(leekrl_env_reset(env, seed) == LEEKRL_ENV_OK);
            for (j = 0; j < length; ++j)
                CHECK(leekrl_env_step(env, history[j], &result) == LEEKRL_ENV_OK);
            history[length] = chosen;
            CHECK(leekrl_env_step(env, chosen, &result) == LEEKRL_ENV_OK);
            if (result.terminated || result.truncated) break;
        }
    }
    leekrl_env_destroy(env);
    leekrl_leekwars_destroy(context);
    return 0;
}

int main(void)
{
    LwEffectParameters effect;
    LwWeapon weapon, short_weapon, weak_weapon;
    leekrl_leekwars *context;
    leekrl_leekwars *invalid = NULL;
    leekrl_env *env;
    leekrl_step_result result;
    float initial[LEEKRL_LW_OBSERVATION_SIZE];
    int i, fired = 0, approaches = 0, fires = 0;

    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_DAMAGE, 150, 150, 0,
                              LW_TARGET_ENEMIES | LW_TARGET_NON_SUMMONS, 0);
    lw_weapon_init(&weapon, 50001, 2, 1, 50, &effect, 1,
                   LW_ATTACK_LAUNCH_TYPE_CIRCLE, 1, 0, 50001,
                   "test weapon", NULL, 0, -1);
    CHECK(lw_weapons_add_weapon(&weapon));
    CHECK(leekrl_leekwars_create(50001, 0, &invalid) == LEEKRL_ENV_INVALID_ARGUMENT);
    CHECK(invalid == NULL);
    CHECK(leekrl_leekwars_create(-1, 3, &invalid) == LEEKRL_ENV_INVALID_ARGUMENT);
    CHECK(leekrl_leekwars_create(50001, 3, &context) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_create(LEEKRL_LW_OBSERVATION_SIZE, LEEKRL_LW_ACTION_COUNT,
                            leekrl_leekwars_callbacks(), context, &env) == LEEKRL_ENV_OK);
    {
        float after_zero[LEEKRL_LW_OBSERVATION_SIZE];
        uint8_t zero_mask[LEEKRL_LW_ACTION_COUNT];
        leekrl_step_result first;
        CHECK(leekrl_env_reset(env, 0) == LEEKRL_ENV_OK);
        CHECK(leekrl_env_step(env, LEEKRL_LW_WAIT, &first) == LEEKRL_ENV_OK);
        memcpy(after_zero, leekrl_env_observation(env), sizeof(after_zero));
        memcpy(zero_mask, leekrl_env_action_mask(env), sizeof(zero_mask));
        CHECK(leekrl_env_reset(env, 0) == LEEKRL_ENV_OK);
        CHECK(leekrl_env_step(env, LEEKRL_LW_WAIT, &result) == LEEKRL_ENV_OK);
        CHECK(result.reward == first.reward && result.terminated == first.terminated &&
              result.truncated == first.truncated);
        CHECK(memcmp(after_zero, leekrl_env_observation(env), sizeof(after_zero)) == 0);
        CHECK(memcmp(zero_mask, leekrl_env_action_mask(env), sizeof(zero_mask)) == 0);
    }
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    memcpy(initial, leekrl_env_observation(env), sizeof(initial));
    CHECK(initial[LEEKRL_LW_SELF_LIFE] == 100);
    CHECK(initial[LEEKRL_LW_ENEMY_LIFE] == 100);
    CHECK(leekrl_env_action_mask(env)[LEEKRL_LW_WAIT]);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    CHECK(memcmp(initial, leekrl_env_observation(env), sizeof(initial)) == 0);
    CHECK(leekrl_env_reset(env, UINT64_C(0x8000007b)) == LEEKRL_ENV_OK);
    CHECK(memcmp(initial, leekrl_env_observation(env), sizeof(initial)) == 0);
    for (i = 0; i < 3; ++i) {
        CHECK(leekrl_env_step(env, LEEKRL_LW_WAIT, &result) == LEEKRL_ENV_OK);
        CHECK(result.reward == 0);
        if (i < 2) CHECK(!result.terminated && !result.truncated);
    }
    CHECK(result.terminated && !result.truncated && result.reward == 0);
    CHECK(leekrl_env_observation(env) != NULL);
    CHECK(leekrl_env_step(env, LEEKRL_LW_WAIT, &result) == LEEKRL_ENV_INVALID_STATE);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    for (i = 0; i < 3; ++i) {
        const uint8_t *mask = leekrl_env_action_mask(env);
        size_t action = mask[LEEKRL_LW_FIRE] ? LEEKRL_LW_FIRE :
                        mask[LEEKRL_LW_APPROACH] ? LEEKRL_LW_APPROACH : LEEKRL_LW_WAIT;
        if (action == LEEKRL_LW_FIRE) fired = 1;
        CHECK(leekrl_env_step(env, action, &result) == LEEKRL_ENV_OK);
        if (result.terminated || result.truncated) break;
    }
    CHECK(fired);
    CHECK(result.terminated && !result.truncated && result.reward == 1);
    CHECK(leekrl_env_observation(env)[LEEKRL_LW_ENEMY_LIFE] <= 0);
    leekrl_env_destroy(env);
    leekrl_leekwars_destroy(context);

    lw_weapon_init(&short_weapon, 50002, 2, 1, 1, &effect, 1,
                   LW_ATTACK_LAUNCH_TYPE_CIRCLE, 1, 0, 50002,
                   "short weapon", NULL, 0, -1);
    CHECK(lw_weapons_add_weapon(&short_weapon));
    CHECK(leekrl_leekwars_create(50002, 32, &context) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_create(LEEKRL_LW_OBSERVATION_SIZE, LEEKRL_LW_ACTION_COUNT,
                            leekrl_leekwars_callbacks(), context, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    CHECK(!leekrl_env_action_mask(env)[LEEKRL_LW_FIRE]);
    CHECK(leekrl_env_action_mask(env)[LEEKRL_LW_APPROACH]);
    CHECK(leekrl_env_step(env, LEEKRL_LW_FIRE, &result) == LEEKRL_ENV_INVALID_ACTION);
    memcpy(initial, leekrl_env_observation(env), sizeof(initial));
    CHECK(leekrl_env_step(env, LEEKRL_LW_APPROACH, &result) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_observation(env)[LEEKRL_LW_SELF_CELL] != initial[LEEKRL_LW_SELF_CELL]);
    {
        float after_move[LEEKRL_LW_OBSERVATION_SIZE];
        uint8_t after_mask[LEEKRL_LW_ACTION_COUNT];
        memcpy(after_move, leekrl_env_observation(env), sizeof(after_move));
        memcpy(after_mask, leekrl_env_action_mask(env), sizeof(after_mask));
        CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
        CHECK(leekrl_env_step(env, LEEKRL_LW_APPROACH, &result) == LEEKRL_ENV_OK);
        CHECK(memcmp(after_move, leekrl_env_observation(env), sizeof(after_move)) == 0);
        CHECK(memcmp(after_mask, leekrl_env_action_mask(env), sizeof(after_mask)) == 0);
    }
    fired = 0;
    for (i = 1; i < 32; ++i) {
        const uint8_t *mask = leekrl_env_action_mask(env);
        size_t action = mask[LEEKRL_LW_FIRE] ? LEEKRL_LW_FIRE :
                        mask[LEEKRL_LW_APPROACH] ? LEEKRL_LW_APPROACH : LEEKRL_LW_WAIT;
        if (action == LEEKRL_LW_FIRE) fired = 1;
        CHECK(leekrl_env_step(env, action, &result) == LEEKRL_ENV_OK);
        if (result.terminated || result.truncated) break;
    }
    CHECK(fired && result.terminated && result.reward == 1);
    leekrl_env_destroy(env);
    leekrl_leekwars_destroy(context);

    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_DAMAGE, 10, 10, 0,
                              LW_TARGET_ENEMIES | LW_TARGET_NON_SUMMONS, 0);
    lw_weapon_init(&weak_weapon, 50003, 2, 1, 50, &effect, 1,
                   LW_ATTACK_LAUNCH_TYPE_CIRCLE, 1, 0, 50003,
                   "weak weapon", NULL, 0, -1);
    CHECK(lw_weapons_add_weapon(&weak_weapon));
    CHECK(leekrl_leekwars_create(50003, 2, &context) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_create(LEEKRL_LW_OBSERVATION_SIZE, LEEKRL_LW_ACTION_COUNT,
                            leekrl_leekwars_callbacks(), context, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_action_mask(env)[LEEKRL_LW_FIRE]);
    CHECK(leekrl_env_step(env, LEEKRL_LW_FIRE, &result) == LEEKRL_ENV_OK);
    CHECK(!result.terminated && !result.truncated && result.reward == 0);
    CHECK(leekrl_env_observation(env)[LEEKRL_LW_ENEMY_LIFE] < 100);
    CHECK(leekrl_env_observation(env)[LEEKRL_LW_ENEMY_LIFE] > 0);
    memcpy(initial, leekrl_env_observation(env), sizeof(initial));
    CHECK(leekrl_env_step(env, LEEKRL_LW_FIRE, &result) == LEEKRL_ENV_OK);
    CHECK(result.terminated && !result.truncated && result.reward == 0);
    CHECK(leekrl_env_observation(env)[LEEKRL_LW_ENEMY_LIFE] <
          initial[LEEKRL_LW_ENEMY_LIFE]);
    CHECK(!leekrl_env_action_mask(env)[LEEKRL_LW_WAIT]);
    CHECK(!leekrl_env_action_mask(env)[LEEKRL_LW_APPROACH]);
    CHECK(!leekrl_env_action_mask(env)[LEEKRL_LW_FIRE]);
    leekrl_env_destroy(env);
    leekrl_leekwars_destroy(context);
    CHECK(exercise_masks(50002, &approaches, &fires) == 0);
    CHECK(exercise_masks(50003, &approaches, &fires) == 0);
    CHECK(approaches > 0 && fires > 0);
    lw_weapons_clear();
    return 0;
}
