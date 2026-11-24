#include "leekrl/battle.h"
#include "leekrl/collect.h"
#include "leekrl/observation.h"
#include "lw_chip.h"
#include "lw_effect_params.h"
#include "lw_weapon.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static void init_entity(LwEntityInfo *info, int id)
{
    lw_entity_info_init(info);
    info->id = id;
    info->life = 100;
    info->tp = 10;
    info->mp = 3;
    info->ram = 10;
    info->level = 1;
}

/* Values of the acting entity, unscaled. */
static int self_value(leekrl_env *env, int offset, int scale)
{
    float value = leekrl_env_observation(env)[LEEKRL_OBS_ENTITIES + offset] * (float)scale;
    return (int)(value + 0.5f);
}

static int turn_of(leekrl_env *env)
{
    return (int)(leekrl_env_observation(env)[LEEKRL_OBS_TURN] * 64 + 0.5f);
}

static size_t find_action(leekrl_env *env, const leekrl_action_catalog *catalog,
                          leekrl_action_kind kind, int id)
{
    const leekrl_action *actions = leekrl_action_catalog_actions(catalog);
    const uint8_t *mask = leekrl_env_action_mask(env);
    size_t i, n = leekrl_action_catalog_count(catalog);
    for (i = 0; i < n; ++i)
        if (mask[i] && actions[i].kind == kind &&
            (kind == LEEKRL_ACTION_MOVE || actions[i].item_id == id)) return i;
    return n;
}

static void check_unequipped(leekrl_env *env, const leekrl_action_catalog *catalog,
                             leekrl_action_kind kind, int id)
{
    const leekrl_action *actions = leekrl_action_catalog_actions(catalog);
    const uint8_t *mask = leekrl_env_action_mask(env);
    size_t i, found = 0, n = leekrl_action_catalog_count(catalog);
    for (i = 0; i < n; ++i) {
        if (actions[i].kind == kind && actions[i].item_id == id) {
            ++found;
            CHECK(!mask[i]);
        }
    }
    CHECK(found != 0);
}

int main(void)
{
    LwEffectParameters effect;
    LwWeapon weapon, other, lethal;
    LwChip chip;
    LwEntityInfo self, opponent;
    leekrl_battle_options options;
    leekrl_battle *battle = NULL;
    leekrl_env *env = NULL;
    const leekrl_action_catalog *catalog;
    leekrl_step_result result;
    float *snapshot;
    uint8_t *mask;
    size_t history[3], n, obs_size, i;
    int turn;

    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_DAMAGE, 1, 1, 0,
                              LW_TARGET_ENEMIES | LW_TARGET_NON_SUMMONS, 0);
    lw_weapon_init(&weapon, 101, 2, 1, 6, &effect, 1, LW_ATTACK_LAUNCH_TYPE_LINE,
                   1, 0, 101, "weapon", NULL, 0, 1);
    lw_weapon_init(&other, 102, 2, 1, 1, &effect, 1, LW_ATTACK_LAUNCH_TYPE_LINE,
                   1, 0, 102, "other", NULL, 0, 1);
    lw_chip_init(&chip, 201, 2, 0, 0, &effect, 1, LW_ATTACK_LAUNCH_TYPE_CIRCLE,
                 1, 0, 2, 0, 0, 1, 201, "chip", LW_CHIP_TYPE_DAMAGE, 1);
    CHECK(lw_weapons_add_weapon(&other) && lw_weapons_add_weapon(&weapon));
    CHECK(lw_chips_add_chip(&chip));
    init_entity(&self, 1);
    init_entity(&opponent, 2);
    self.weapons[0] = 101;
    self.n_weapons = 1;
    self.chips[0] = 201;
    self.n_chips = 1;
    leekrl_battle_options_init(&options);
    options.max_turns = 3;
    CHECK(leekrl_battle_create(&self, &opponent, &options, &battle) == LEEKRL_ENV_OK);
    catalog = leekrl_battle_catalog(battle);
    n = leekrl_action_catalog_count(catalog);
    obs_size = leekrl_battle_observation_size(battle);
    CHECK(n > 5 && obs_size == LEEKRL_OBS_ITEMS + 2 * (2 * 2 + 2 * 1) + 3 +
                               LEEKRL_OBS_MAP_CELLS);
    CHECK(leekrl_env_create(obs_size, n, leekrl_battle_callbacks(), battle, &env) ==
          LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_observation(env)[LEEKRL_OBS_ENTITIES + LEEKRL_OBS_LIFE] == 1);
    CHECK(leekrl_battle_life(battle, 0) == 100 && leekrl_battle_life(battle, 1) == 100);
    CHECK(leekrl_env_action_mask(env)[0]);
    check_unequipped(env, catalog, LEEKRL_ACTION_WEAPON, 102);
    {
        leekrl_battle *other_battle = NULL;
        leekrl_env *other_env = NULL;
        const leekrl_action_catalog *other_catalog;
        const leekrl_action *a = leekrl_action_catalog_actions(catalog), *b;
        self.weapons[0] = 102;
        self.n_chips = 0;
        CHECK(leekrl_battle_create(&self, &opponent, &options, &other_battle) == LEEKRL_ENV_OK);
        other_catalog = leekrl_battle_catalog(other_battle);
        CHECK(leekrl_action_catalog_count(other_catalog) == n);
        b = leekrl_action_catalog_actions(other_catalog);
        for (i = 0; i < n; ++i)
            CHECK(a[i].kind == b[i].kind && a[i].item_id == b[i].item_id &&
                  a[i].dx == b[i].dx && a[i].dy == b[i].dy);
        CHECK(leekrl_env_create(leekrl_battle_observation_size(other_battle), n,
                                leekrl_battle_callbacks(), other_battle, &other_env) ==
              LEEKRL_ENV_OK);
        CHECK(leekrl_env_reset(other_env, 123) == LEEKRL_ENV_OK);
        check_unequipped(other_env, other_catalog, LEEKRL_ACTION_WEAPON, 101);
        check_unequipped(other_env, other_catalog, LEEKRL_ACTION_CHIP, 201);
        CHECK(find_action(other_env, other_catalog, LEEKRL_ACTION_WEAPON, 102) < n);
        leekrl_env_destroy(other_env);
        leekrl_battle_destroy(other_battle);
        self.weapons[0] = 101;
        self.n_chips = 1;
    }
    turn = turn_of(env);
    history[0] = find_action(env, catalog, LEEKRL_ACTION_MOVE, 0);
    CHECK(history[0] < n);
    CHECK(leekrl_env_step(env, history[0], &result) == LEEKRL_ENV_OK);
    CHECK(!result.terminated && !result.truncated && result.reward == 0);
    CHECK(turn_of(env) == turn);
    CHECK(self_value(env, LEEKRL_OBS_MP, 15) == 2);
    history[1] = find_action(env, catalog, LEEKRL_ACTION_WEAPON, 101);
    CHECK(history[1] < n);
    CHECK(leekrl_env_step(env, history[1], &result) == LEEKRL_ENV_OK);
    CHECK(!result.terminated && turn_of(env) == turn);
    CHECK(self_value(env, LEEKRL_OBS_TP, 30) == 7);
    history[2] = find_action(env, catalog, LEEKRL_ACTION_CHIP, 201);
    CHECK(history[2] < n);
    CHECK(leekrl_env_step(env, history[2], &result) == LEEKRL_ENV_OK);
    CHECK(!result.terminated && turn_of(env) == turn);
    CHECK(self_value(env, LEEKRL_OBS_TP, 30) == 5);
    CHECK(!leekrl_env_action_mask(env)[history[2]]);
    snapshot = malloc(obs_size * sizeof(*snapshot));
    mask = malloc(n);
    CHECK(snapshot && mask);
    memcpy(snapshot, leekrl_env_observation(env), obs_size * sizeof(*snapshot));
    memcpy(mask, leekrl_env_action_mask(env), n);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    for (i = 0; i < 3; ++i) CHECK(leekrl_env_step(env, history[i], &result) == LEEKRL_ENV_OK);
    CHECK(memcmp(snapshot, leekrl_env_observation(env), obs_size * sizeof(*snapshot)) == 0);
    CHECK(memcmp(mask, leekrl_env_action_mask(env), n) == 0);
    CHECK(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_OK);
    CHECK(!result.terminated && turn_of(env) > turn);
    CHECK(self_value(env, LEEKRL_OBS_MP, 15) == 3);
    {
        lr_ppo *model = NULL;
        lr_trajectory *trajectory = NULL;
        leekrl_collector *collector = NULL;
        leekrl_collect_result collected;
        lr_ppo_options ppo_options = {0.01, 0.2, 0.5, 0.01, 2};
        lr_rng rng;
        CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
        lr_rng_seed(&rng, 2);
        CHECK(lr_ppo_create(&model, obs_size, 8, n, &rng) == LR_OK);
        CHECK(lr_trajectory_create(&trajectory, 4, obs_size, n) == LR_OK);
        CHECK(leekrl_collector_create(env, model, 3, 124, &collector) == LEEKRL_COLLECT_OK);
        CHECK(leekrl_collector_collect(collector, trajectory, 4, &collected) ==
              LEEKRL_COLLECT_OK);
        CHECK(collected.steps_collected == 4);
        CHECK(lr_trajectory_gae(trajectory, 0.99, 0.95) == LR_OK);
        CHECK(lr_ppo_update(model, trajectory, &ppo_options) == LR_OK);
        leekrl_collector_destroy(collector);
        lr_trajectory_destroy(trajectory);
        lr_ppo_destroy(model);
    }
    free(mask);
    free(snapshot);
    leekrl_env_destroy(env);
    leekrl_battle_destroy(battle);

    options.max_decisions = 1;
    options.step_reward = 0.25f;
    CHECK(leekrl_battle_create(&self, &opponent, &options, &battle) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_create(leekrl_battle_observation_size(battle),
                            leekrl_action_catalog_count(leekrl_battle_catalog(battle)),
                            leekrl_battle_callbacks(), battle, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_OK);
    CHECK(!result.terminated && result.truncated && result.reward == 0.25f);
    CHECK(leekrl_env_action_mask(env)[0]);
    leekrl_env_destroy(env);
    leekrl_battle_destroy(battle);

    options.max_decisions = 10;
    options.max_turns = 1;
    options.step_reward = 0;
    options.draw_reward = 0.5f;
    CHECK(leekrl_battle_create(&self, &opponent, &options, &battle) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_create(leekrl_battle_observation_size(battle),
                            leekrl_action_catalog_count(leekrl_battle_catalog(battle)),
                            leekrl_battle_callbacks(), battle, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_OK);
    CHECK(result.terminated && !result.truncated && result.reward == 0.5f);
    CHECK(!leekrl_env_action_mask(env)[0]);
    leekrl_env_destroy(env);
    leekrl_battle_destroy(battle);

    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_DAMAGE, 150, 150, 0,
                              LW_TARGET_ENEMIES | LW_TARGET_NON_SUMMONS, 0);
    lw_weapon_init(&lethal, 103, 2, 1, 50, &effect, 1, LW_ATTACK_LAUNCH_TYPE_CIRCLE,
                   1, 0, 103, "lethal", NULL, 0, -1);
    CHECK(lw_weapons_add_weapon(&lethal));
    self.weapons[0] = 103;
    self.n_chips = 0;
    options.max_decisions = 10;
    options.max_turns = 2;
    options.win_reward = 2;
    CHECK(leekrl_battle_create(&self, &opponent, &options, &battle) == LEEKRL_ENV_OK);
    catalog = leekrl_battle_catalog(battle);
    n = leekrl_action_catalog_count(catalog);
    CHECK(leekrl_env_create(leekrl_battle_observation_size(battle), n,
                            leekrl_battle_callbacks(), battle, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    {
        const float *entity = leekrl_env_observation(env) + LEEKRL_OBS_ENTITIES +
                              LEEKRL_OBS_ENTITY_SIZE;
        const leekrl_action *actions = leekrl_action_catalog_actions(catalog);
        const uint8_t *legal = leekrl_env_action_mask(env);
        size_t selected = n;
        int dx = (int)lround(entity[LEEKRL_OBS_DX] * LEEKRL_OBS_MAP_RADIUS);
        int dy = (int)lround(entity[LEEKRL_OBS_DY] * LEEKRL_OBS_MAP_RADIUS);
        CHECK(entity[LEEKRL_OBS_PRESENT] == 1 && entity[LEEKRL_OBS_ALLY] == 0);
        CHECK(leekrl_env_observation(env)[leekrl_battle_observation_size(battle) -
              LEEKRL_OBS_MAP_CELLS + (size_t)leekrl_obs_map_index(dx, dy)] ==
              LEEKRL_CELL_ENEMY_LEEK);
        for (i = 0; i < n; ++i) {
            if (actions[i].kind == LEEKRL_ACTION_WEAPON && actions[i].item_id == 103 &&
                actions[i].dx == dx && actions[i].dy == dy && legal[i]) {
                selected = i;
                break;
            }
        }
        CHECK(selected < n);
        CHECK(leekrl_env_step(env, selected, &result) == LEEKRL_ENV_OK);
        CHECK(result.terminated && !result.truncated && result.reward == 2);
        CHECK(leekrl_battle_life(battle, 1) <= 0 && leekrl_battle_life(battle, 0) == 100);
        CHECK(leekrl_env_observation(env)[LEEKRL_OBS_ENTITIES + LEEKRL_OBS_PRESENT] == 0);
        CHECK(!leekrl_env_action_mask(env)[0]);
    }
    leekrl_env_destroy(env);
    leekrl_battle_destroy(battle);
    lw_weapons_clear();
    lw_chips_clear();
    return 0;
}
