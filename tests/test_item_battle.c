#include "leekrl/actions.h"
#include "leekrl/battle.h"
#include "leekrl/collect.h"
#include "leekrl/grid.h"
#include "leekrl/item_snapshot.h"
#include "lw_bulb_template.h"
#include "lw_chip.h"
#include "lw_weapon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static void init_entity(LwEntityInfo *entity, int id)
{
    lw_entity_info_init(entity);
    entity->id = id;
    entity->life = 100;
    entity->tp = 10;
    entity->mp = 3;
    entity->ram = 10;
    entity->level = 1;
}

static void check_registered_items(void)
{
    LwWeapon *weapon = lw_weapons_get_weapon(108);
    LwWeapon *passive_weapon = lw_weapons_get_weapon(226);
    LwChip *chip = lw_chips_get_chip(85);
    LwBulbTemplate *summon = lw_bulbs_get_invocation_template(1);
    int weapons, chips;
    CHECK(lw_weapons_get_templates(&weapons) && weapons == 40);
    CHECK(lw_chips_get_templates(&chips) && chips == 121);
    CHECK(weapon && lw_weapon_get_cost(weapon) == 5 &&
          lw_attack_get_min_range(lw_weapon_get_attack(weapon)) == 1 &&
          lw_attack_get_max_range(lw_weapon_get_attack(weapon)) == 1 &&
          lw_attack_get_max_uses(lw_weapon_get_attack(weapon)) == 2);
    CHECK(passive_weapon && lw_weapon_get_passive_effects_count(passive_weapon) == 1);
    CHECK(chip && lw_chip_get_cost(chip) == 6 && lw_chip_get_cooldown(chip) == 3 &&
          lw_attack_get_min_range(lw_chip_get_attack(chip)) == 0 &&
          lw_attack_get_max_range(lw_chip_get_attack(chip)) == 0 &&
          lw_attack_get_effects_count(lw_chip_get_attack(chip)) == 5);
    CHECK(summon && lw_bulb_template_get_min_life(summon) == 50 &&
          lw_bulb_template_get_max_life(summon) == 300 &&
          lw_bulb_template_get_chips_count(summon) == 4);
    CHECK(lw_bulbs_get_invocation_template(13));
}

static void check_build(void)
{
    LwEntityInfo self, opponent;
    leekrl_battle_options options;
    leekrl_battle *battle = NULL;
    leekrl_env *env = NULL;
    leekrl_step_result result;
    const leekrl_action_catalog *catalog;
    const leekrl_action *actions;
    const uint8_t *mask;
    size_t i, n, devil_index = 0, devil_count = 0, sword_count = 0, sword_legal = 0;
    init_entity(&self, 1);
    init_entity(&opponent, 2);
    self.life = 500;
    self.weapons[0] = 108;
    self.n_weapons = 1;
    self.chips[0] = 85;
    self.n_chips = 1;
    leekrl_battle_options_init(&options);
    options.max_turns = 2;
    options.grid_radius = 4;
    CHECK(leekrl_battle_create(&self, &opponent, &options, &battle) == LEEKRL_ENV_OK);
    catalog = leekrl_battle_catalog(battle);
    n = leekrl_action_catalog_count(catalog);
    CHECK(n == 31285);
    actions = leekrl_action_catalog_actions(catalog);
    CHECK(leekrl_env_create(leekrl_battle_observation_size(battle), n,
                            leekrl_battle_callbacks(), battle, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    mask = leekrl_env_action_mask(env);
    for (i = 0; i < n; ++i) {
        if (actions[i].kind == LEEKRL_ACTION_CHIP && actions[i].item_id == 85) {
            ++devil_count;
            devil_index = i;
            CHECK(actions[i].dx == 0 && actions[i].dy == 0 && mask[i]);
        } else if (actions[i].kind == LEEKRL_ACTION_WEAPON && actions[i].item_id == 108) {
            ++sword_count;
            CHECK(abs(actions[i].dx) + abs(actions[i].dy) == 1);
            sword_legal += mask[i];
        } else if (actions[i].kind == LEEKRL_ACTION_CHIP ||
                   actions[i].kind == LEEKRL_ACTION_WEAPON) {
            CHECK(!mask[i]);
        }
    }
    CHECK(devil_count == 1 && sword_count == 4 && sword_legal > 0);
    CHECK(leekrl_env_step(env, devil_index, &result) == LEEKRL_ENV_OK);
    CHECK(!result.terminated && !result.truncated);
    CHECK(leekrl_env_observation(env)[LEEKRL_BATTLE_TP] == 4);
    mask = leekrl_env_action_mask(env);
    CHECK(!mask[devil_index]);
    for (i = 0; i < n; ++i)
        if (actions[i].kind == LEEKRL_ACTION_WEAPON && actions[i].item_id == 108)
            CHECK(!mask[i]);
    {
        lr_ppo *model = NULL;
        lr_trajectory *trajectory = NULL;
        leekrl_collector *collector = NULL;
        leekrl_collect_result collected;
        lr_ppo_options ppo_options = {0.01, 0.2, 0.5, 0.01, 1};
        lr_rng rng;
        size_t obs_size = leekrl_battle_observation_size(battle);
        CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
        lr_rng_seed(&rng, 2);
        CHECK(lr_ppo_create(&model, obs_size, 8, n, &rng) == LR_OK);
        CHECK(lr_trajectory_create(&trajectory, 2, obs_size, n) == LR_OK);
        CHECK(leekrl_collector_create(env, model, 3, 124, &collector) == LEEKRL_COLLECT_OK);
        CHECK(leekrl_collector_collect(collector, trajectory, 2, &collected) ==
              LEEKRL_COLLECT_OK && collected.steps_collected == 2);
        CHECK(lr_trajectory_gae(trajectory, 0.99, 0.95) == LR_OK);
        CHECK(lr_ppo_update(model, trajectory, &ppo_options) == LR_OK);
        leekrl_collector_destroy(collector);
        lr_trajectory_destroy(trajectory);
        lr_ppo_destroy(model);
    }
    leekrl_env_destroy(env);
    leekrl_battle_destroy(battle);
}

static void check_summon(void)
{
    LwEntityInfo self, opponent;
    leekrl_battle_options options;
    leekrl_battle *battle = NULL;
    leekrl_env *env = NULL;
    leekrl_step_result result;
    const leekrl_action_catalog *catalog;
    const leekrl_action *actions;
    const uint8_t *mask;
    const float *grid;
    size_t i, n, selected, allies = 0, side;
    CHECK(lw_bulbs_get_invocation_template(1));
    init_entity(&self, 1);
    init_entity(&opponent, 2);
    self.chips[0] = 73;
    self.n_chips = 1;
    leekrl_battle_options_init(&options);
    options.max_turns = 2;
    options.grid_radius = 4;
    CHECK(leekrl_battle_create(&self, &opponent, &options, &battle) == LEEKRL_ENV_OK);
    catalog = leekrl_battle_catalog(battle);
    n = leekrl_action_catalog_count(catalog);
    actions = leekrl_action_catalog_actions(catalog);
    CHECK(leekrl_env_create(leekrl_battle_observation_size(battle), n,
                            leekrl_battle_callbacks(), battle, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 123) == LEEKRL_ENV_OK);
    mask = leekrl_env_action_mask(env);
    for (selected = 0; selected < n; ++selected)
        if (actions[selected].kind == LEEKRL_ACTION_CHIP &&
            actions[selected].item_id == 73 && mask[selected]) break;
    CHECK(selected < n);
    CHECK(leekrl_env_step(env, selected, &result) == LEEKRL_ENV_OK);
    CHECK(!result.terminated && !result.truncated);
    CHECK(!leekrl_env_action_mask(env)[selected]);
    grid = leekrl_env_observation(env) + LEEKRL_BATTLE_GRID;
    side = (size_t)(2 * options.grid_radius + 1);
    for (i = 0; i < side * side; ++i)
        allies += grid[LEEKRL_GRID_ALLY * side * side + i] == 1;
    CHECK(allies == 1);
    leekrl_env_destroy(env);
    leekrl_battle_destroy(battle);
}

int main(void)
{
    char error[512];
    LwWeapon borrowed;
    lw_weapon_init(&borrowed, 999, 1, 1, 1, NULL, 0, 7, 1, 1,
                   999, "borrowed", NULL, 0, 1);
    CHECK(lw_weapons_add_weapon(&borrowed));
    CHECK(!leekrl_item_snapshot_register(error, sizeof(error)));
    CHECK(error[0] && lw_weapons_get_weapon(999) == &borrowed);
    lw_weapons_clear();
    CHECK(strlen(leekrl_item_snapshot_hash()) == 64);
    if (!leekrl_item_snapshot_register(error, sizeof(error))) {
        fprintf(stderr, "item registration: %s\n", error);
        return 1;
    }
    CHECK(leekrl_item_snapshot_register(error, sizeof(error)));
    check_registered_items();
    check_build();
    check_summon();
    leekrl_item_snapshot_clear();
    CHECK(!lw_weapons_get_weapon(108));
    CHECK(!lw_chips_get_chip(85));
    CHECK(!lw_bulbs_get_invocation_template(1));
    CHECK(leekrl_item_snapshot_register(error, sizeof(error)));
    CHECK(lw_weapons_get_weapon(108) && lw_bulbs_get_invocation_template(1));
    leekrl_item_snapshot_clear();
    return 0;
}
