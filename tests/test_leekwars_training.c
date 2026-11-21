#include "leekrl/collect.h"
#include "leekrl/leekwars.h"
#include "lw_constants.h"
#include "lw_effect_params.h"
#include "lw_weapon.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

int main(void)
{
    LwEffectParameters effect;
    LwWeapon weapon;
    leekrl_leekwars *context = NULL;
    leekrl_env *env = NULL;
    leekrl_collector *collector = NULL;
    leekrl_collect_result result;
    lr_ppo *model = NULL;
    lr_trajectory *trajectory = NULL;
    lr_rng rng;
    lr_ppo_options options = {0.01, 0.2, 0.5, 0.01, 2};
    int i;

    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_DAMAGE, 150, 150, 0,
                              LW_TARGET_ENEMIES | LW_TARGET_NON_SUMMONS, 0);
    lw_weapon_init(&weapon, 50010, 2, 1, 50, &effect, 1,
                   LW_ATTACK_LAUNCH_TYPE_CIRCLE, 1, 0, 50010,
                   "test weapon", NULL, 0, -1);
    CHECK(lw_weapons_add_weapon(&weapon));
    CHECK(leekrl_leekwars_create(50010, 3, &context) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_create(LEEKRL_LW_OBSERVATION_SIZE, LEEKRL_LW_ACTION_COUNT,
                            leekrl_leekwars_callbacks(), context, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    lr_rng_seed(&rng, 2);
    CHECK(lr_ppo_create(&model, LEEKRL_LW_OBSERVATION_SIZE, 8,
                        LEEKRL_LW_ACTION_COUNT, &rng) == LR_OK);
    CHECK(lr_trajectory_create(&trajectory, 12, LEEKRL_LW_OBSERVATION_SIZE,
                               LEEKRL_LW_ACTION_COUNT) == LR_OK);
    CHECK(leekrl_collector_create(env, model, 3, 124, &collector) == LEEKRL_COLLECT_OK);
    for (i = 0; i < 2; ++i) {
        lr_trajectory_clear(trajectory);
        CHECK(leekrl_collector_collect(collector, trajectory, 12, &result) ==
              LEEKRL_COLLECT_OK);
        CHECK(result.steps_collected == 12);
        CHECK(lr_trajectory_gae(trajectory, 0.99, 0.95) == LR_OK);
        CHECK(lr_ppo_update(model, trajectory, &options) == LR_OK);
    }
    leekrl_collector_destroy(collector);
    lr_trajectory_destroy(trajectory);
    lr_ppo_destroy(model);
    leekrl_env_destroy(env);
    leekrl_leekwars_destroy(context);
    lw_weapons_clear();
    return 0;
}
