#include "leekrl/actions.h"

#include "lw_chip.h"
#include "lw_bulb_template.h"
#include "lw_effect_params.h"
#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_generator.h"
#include "lw_map.h"
#include "lw_state.h"
#include "lw_stats.h"
#include "lw_team.h"
#include "lw_weapon.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

typedef struct {
    LwState state;
    LwMap map;
    LwFight fight;
    LwEntity actor;
    LwEntity enemy;
    LwTeam teams[2];
    LwWeapon weapon;
    LwChip chip;
    LwChip teleport;
} fixture;

static fixture *make_fixture(void)
{
    fixture *f = calloc(1, sizeof(*f));
    LwEffectParameters effect;
    CHECK(f);
    lw_state_init(&f->state);
    lw_map_init(&f->map, 18, 18);
    f->map.state = &f->state;
    f->state.map = &f->map;
    lw_fight_init(&f->fight, NULL);
    lw_fight_set_state(&f->fight, &f->state);
    lw_entity_init_id_name(&f->actor, 1, "actor");
    lw_entity_init_id_name(&f->enemy, 2, "enemy");
    lw_entity_set_state(&f->actor, &f->state, 0);
    lw_entity_set_state(&f->enemy, &f->state, 1);
    lw_entity_set_team(&f->actor, 0);
    lw_entity_set_team(&f->enemy, 1);
    lw_team_init(&f->teams[0]);
    lw_team_init(&f->teams[1]);
    f->state.teams[0] = &f->teams[0];
    f->state.teams[1] = &f->teams[1];
    f->state.n_teams = 2;
    lw_team_add_entity(&f->teams[0], &f->actor);
    lw_team_add_entity(&f->teams[1], &f->enemy);
    f->actor.life = f->actor.m_total_life = 100;
    f->enemy.life = f->enemy.m_total_life = 100;
    lw_stats_set_stat(&f->actor.m_base_stats, LW_STAT_TP, 10);
    lw_stats_set_stat(&f->actor.m_base_stats, LW_STAT_MP, 3);
    lw_stats_set_stat(&f->actor.m_base_stats, LW_STAT_RAM, 10);
    f->state.m_entities[f->state.n_entities++] = &f->actor;
    f->state.m_entities[f->state.n_entities++] = &f->enemy;
    lw_order_add_entity(&f->state.order, &f->actor);
    lw_map_set_entity(&f->map, &f->actor, lw_map_get_cell(&f->map, 306));
    lw_map_set_entity(&f->map, &f->enemy, lw_map_get_cell(&f->map, 309));

    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_DAMAGE, 1, 1, 0,
                              LW_TARGET_ENEMIES | LW_TARGET_NON_SUMMONS, 0);
    lw_weapon_init(&f->weapon, 101, 2, 1, 6, &effect, 1,
                   LW_ATTACK_LAUNCH_TYPE_LINE, 1, 1, 101, "line", NULL, 0, 1);
    lw_chip_init(&f->chip, 201, 2, 0, 6, &effect, 1,
                 LW_ATTACK_LAUNCH_TYPE_CIRCLE, 1, 1, 2, 0, 0,
                 1, 201, "chip", LW_CHIP_TYPE_DAMAGE, 1);
    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_TELEPORT, 0, 0, 0, 0, 0);
    lw_chip_init(&f->teleport, 202, 2, 1, 6, &effect, 1,
                 LW_ATTACK_LAUNCH_TYPE_CIRCLE, 1, 1, 0, 0, 0,
                 1, 202, "teleport", LW_CHIP_TYPE_TACTIC, -1);
    lw_entity_add_weapon(&f->actor, &f->weapon);
    lw_entity_add_chip(&f->actor, &f->chip);
    lw_entity_add_chip(&f->actor, &f->teleport);
    return f;
}

static void free_fixture(fixture *f)
{
    lw_state_dispose(&f->state);
    lw_entity_dispose(&f->actor);
    lw_entity_dispose(&f->enemy);
    free(f);
}

static uint8_t mask_one(fixture *f, leekrl_action action)
{
    uint8_t mask = 9;
    CHECK(leekrl_action_mask(&f->fight, &f->actor, &action, 1, &mask) ==
          LEEKRL_ACTION_OK);
    CHECK(mask <= 1);
    return mask;
}

static void test_line_mask_against_engine(void)
{
    fixture *layout_fixture = make_fixture();
    leekrl_action layout[24], short_buffer[1] = {{LEEKRL_ACTION_END_TURN, 0, 0, 0}};
    size_t i, required = 0;
    int valid = 0;
    CHECK(leekrl_action_cast_layout(lw_weapon_get_attack(&layout_fixture->weapon),
                                    LEEKRL_ACTION_WEAPON, 101, NULL, 0, &required) ==
          LEEKRL_ACTION_OK);
    CHECK(required == 24);
    CHECK(leekrl_action_cast_layout(lw_weapon_get_attack(&layout_fixture->weapon),
                                    LEEKRL_ACTION_WEAPON, 101, short_buffer, 1, &required) ==
          LEEKRL_ACTION_BUFFER_TOO_SMALL);
    CHECK(required == 24 && short_buffer[0].kind == LEEKRL_ACTION_END_TURN);
    CHECK(leekrl_action_cast_layout(lw_weapon_get_attack(&layout_fixture->weapon),
                                    LEEKRL_ACTION_WEAPON, 101, layout, 24, &required) ==
          LEEKRL_ACTION_OK);
    for (i = 0; i < required; ++i) {
            fixture *f = make_fixture();
            leekrl_action action = layout[i];
            LwCell *target = lw_cell_next(lw_entity_get_cell(&f->actor), &f->map,
                                          action.dx, action.dy);
            uint8_t mask = mask_one(f, action);
            int engine_result = LW_ATTACK_USE_INVALID_POSITION;
            CHECK(action.kind == LEEKRL_ACTION_WEAPON && action.item_id == 101);
            CHECK((action.dx == 0) != (action.dy == 0));
            CHECK(abs(action.dx) + abs(action.dy) >= 1);
            CHECK(abs(action.dx) + abs(action.dy) <= 6);
            if (target && lw_state_set_weapon(&f->state, &f->actor, &f->weapon))
                engine_result = lw_state_use_weapon(&f->state, &f->actor, target);
            CHECK(mask == (engine_result > 0));
            if (mask) ++valid;
            free_fixture(f);
    }
    CHECK(valid == 24);
    free_fixture(layout_fixture);
}

static void test_chip_mask_against_engine(void)
{
    int dx, dy, valid = 0, invalid = 0;
    for (dy = -7; dy <= 7; ++dy) {
        for (dx = -7; dx <= 7; ++dx) {
            fixture *f = make_fixture();
            leekrl_action action = {LEEKRL_ACTION_CHIP, 201, dx, dy};
            LwCell *target = lw_cell_next(lw_entity_get_cell(&f->actor), &f->map, dx, dy);
            uint8_t mask = mask_one(f, action);
            int engine_result = LW_ATTACK_USE_INVALID_POSITION;
            if (target) engine_result = lw_fight_use_chip(&f->fight, &f->actor, target, &f->chip);
            CHECK(mask == (engine_result > 0));
            if (mask) ++valid; else ++invalid;
            free_fixture(f);
        }
    }
    CHECK(valid > 0 && invalid > 0);
}

static void test_masks_change_with_state(void)
{
    fixture *f = make_fixture();
    leekrl_action actions[] = {
        {LEEKRL_ACTION_MOVE, 0, 1, 0},
        {LEEKRL_ACTION_WEAPON, 101, 1, 0},
        {LEEKRL_ACTION_CHIP, 201, 0, 0},
        {LEEKRL_ACTION_CHIP, 202, 1, 0},
        {LEEKRL_ACTION_WEAPON, 999, 1, 0},
        {LEEKRL_ACTION_CHIP, 999, 0, 0},
        {LEEKRL_ACTION_END_TURN, 0, 0, 0}
    };
    uint8_t mask[7];
    LwCell *from = lw_entity_get_cell(&f->actor);
    LwCell *next = lw_cell_next(from, &f->map, 1, 0);
    int result;
    CHECK(next && lw_cell_available(next, &f->map));
    CHECK(leekrl_action_mask(&f->fight, &f->actor, actions, 7, mask) == LEEKRL_ACTION_OK);
    CHECK(mask[0] && mask[1] && mask[2] && mask[3]);
    CHECK(!mask[4] && !mask[5] && mask[6]);
    CHECK(leekrl_action_execute(&f->fight, &f->actor, &actions[2], &result) ==
          LEEKRL_ACTION_OK);
    CHECK(result > 0 && !mask_one(f, actions[2]));

    lw_entity_set_tp(&f->actor, 0);
    CHECK(!mask_one(f, actions[1]) && !mask_one(f, actions[3]));
    lw_entity_set_tp(&f->actor, 10);
    CHECK(leekrl_action_execute(&f->fight, &f->actor, &actions[1], &result) ==
          LEEKRL_ACTION_OK);
    CHECK(result > 0 && !mask_one(f, actions[1]));
    CHECK(leekrl_action_execute(&f->fight, &f->actor, &actions[1], &result) ==
          LEEKRL_ACTION_ILLEGAL);
    lw_cell_set_obstacle(next, 1, 1);
    CHECK(!mask_one(f, actions[0]) && !mask_one(f, actions[3]));
    CHECK(leekrl_action_execute(&f->fight, &f->actor, &actions[6], &result) ==
          LEEKRL_ACTION_OK && result == 0);
    free_fixture(f);
}

static void test_movement_and_argument_errors(void)
{
    fixture *f = make_fixture();
    leekrl_action move = {LEEKRL_ACTION_MOVE, 0, 1, 0};
    leekrl_action invalid = {LEEKRL_ACTION_MOVE, 0, 21, 0};
    LwCell *target = lw_cell_next(lw_entity_get_cell(&f->actor), &f->map, 1, 0);
    int result;
    uint8_t mask;
    CHECK(mask_one(f, move));
    CHECK(leekrl_action_execute(&f->fight, &f->actor, &move, &result) == LEEKRL_ACTION_OK);
    CHECK(result == 1 && lw_entity_get_cell(&f->actor) == target);
    CHECK(leekrl_action_mask(&f->fight, &f->actor, &invalid, 1, &mask) ==
          LEEKRL_ACTION_INVALID_ARGUMENT);
    CHECK(leekrl_action_mask(&f->fight, &f->enemy, &move, 1, &mask) ==
          LEEKRL_ACTION_INVALID_STATE);
    CHECK(leekrl_action_execute(&f->fight, &f->actor, &move, NULL) ==
          LEEKRL_ACTION_INVALID_ARGUMENT);
    lw_entity_set_mp(&f->actor, 0);
    CHECK(!mask_one(f, move));
    free_fixture(f);
}

static void test_long_movement(void)
{
    fixture *f = make_fixture();
    LwCell *start = lw_entity_get_cell(&f->actor);
    LwCell *obstacle = lw_cell_next(start, &f->map, 1, 0);
    LwCell *target = lw_cell_next(start, &f->map, 2, 0);
    leekrl_action move = {LEEKRL_ACTION_MOVE, 0, 2, 0};
    int result, i, found = 0;
    CHECK(obstacle && target && lw_cell_available(target, &f->map));
    lw_cell_set_obstacle(obstacle, 1, 1);
    CHECK(!mask_one(f, move));
    lw_entity_set_mp(&f->actor, 4);
    CHECK(mask_one(f, move));
    CHECK(leekrl_action_execute(&f->fight, &f->actor, &move, &result) == LEEKRL_ACTION_OK);
    CHECK(result == 4 && lw_entity_get_cell(&f->actor) == target);
    CHECK(lw_entity_get_mp(&f->actor) == 0);
    free_fixture(f);

    f = make_fixture();
    start = lw_entity_get_cell(&f->actor);
    {
        LwCell *enemy_cell = lw_entity_get_cell(&f->enemy);
        move.dx = lw_cell_get_x(enemy_cell) - lw_cell_get_x(start);
        move.dy = lw_cell_get_y(enemy_cell) - lw_cell_get_y(start);
        lw_entity_set_mp(&f->actor, 20);
        CHECK(!mask_one(f, move));
    }
    free_fixture(f);

    f = make_fixture();
    lw_map_set_entity(&f->map, &f->actor, lw_map_get_cell(&f->map, 0));
    lw_entity_set_mp(&f->actor, 21);
    start = lw_entity_get_cell(&f->actor);
    for (i = 0; i < f->map.nb_cells; ++i) {
        LwCell *path[LW_MAP_MAX_CELLS];
        LwCell *cell = lw_map_get_cell(&f->map, i);
        int dx = lw_cell_get_x(cell) - lw_cell_get_x(start);
        int dy = lw_cell_get_y(cell) - lw_cell_get_y(start);
        if (abs(dx) + abs(dy) != 20 || !lw_cell_available(cell, &f->map)) continue;
        if (lw_map_get_path_between(&f->map, start, cell, path, LW_MAP_MAX_CELLS) != 20)
            continue;
        move.dx = dx;
        move.dy = dy;
        CHECK(mask_one(f, move));
        CHECK(leekrl_action_execute(&f->fight, &f->actor, &move, &result) ==
              LEEKRL_ACTION_OK);
        CHECK(result == 20 && lw_entity_get_cell(&f->actor) == cell);
        CHECK(lw_entity_get_mp(&f->actor) == 1);
        {
            leekrl_action adjacent[] = {
                {LEEKRL_ACTION_MOVE, 0, 0, -1}, {LEEKRL_ACTION_MOVE, 0, -1, 0},
                {LEEKRL_ACTION_MOVE, 0, 1, 0}, {LEEKRL_ACTION_MOVE, 0, 0, 1}
            };
            uint8_t legal[4];
            int j;
            CHECK(leekrl_action_mask(&f->fight, &f->actor, adjacent, 4, legal) ==
                  LEEKRL_ACTION_OK);
            for (j = 0; j < 4 && !legal[j]; ++j) {}
            CHECK(j < 4);
            CHECK(leekrl_action_execute(&f->fight, &f->actor, &adjacent[j], &result) ==
                  LEEKRL_ACTION_OK);
            CHECK(result == 1 && lw_entity_get_mp(&f->actor) == 0);
        }
        found = 1;
        break;
    }
    CHECK(found);
    free_fixture(f);
}

static void test_obstacles_and_states(void)
{
    fixture *f = make_fixture();
    leekrl_action fire = {LEEKRL_ACTION_WEAPON, 101, 2, 0};
    leekrl_action move = {LEEKRL_ACTION_MOVE, 0, 1, 0};
    leekrl_action teleport = {LEEKRL_ACTION_CHIP, 202, 1, 0};
    LwCell *start = lw_entity_get_cell(&f->actor);
    LwCell *obstacle = lw_cell_next(start, &f->map, 1, 0);
    LwCell *target = lw_cell_next(start, &f->map, 2, 0);
    CHECK(obstacle && target && mask_one(f, fire));
    lw_cell_set_obstacle(obstacle, 1, 1);
    CHECK(!mask_one(f, fire) && !mask_one(f, move) && !mask_one(f, teleport));
    CHECK(lw_state_set_weapon(&f->state, &f->actor, &f->weapon));
    CHECK(lw_state_use_weapon(&f->state, &f->actor, target) ==
          LW_ATTACK_USE_INVALID_POSITION);
    free_fixture(f);

    f = make_fixture();
    CHECK(mask_one(f, move) && mask_one(f, fire));
    lw_entity_add_state(&f->actor, LW_ENTITY_STATE_ROOTED);
    CHECK(!mask_one(f, move));
    lw_entity_add_state(&f->actor, LW_ENTITY_STATE_AMPUTATED);
    CHECK(!mask_one(f, fire));
    free_fixture(f);
}

static void test_chip_limits(void)
{
    fixture *f = make_fixture();
    leekrl_action chip = {LEEKRL_ACTION_CHIP, 201, 0, 0};
    leekrl_action extreme = {LEEKRL_ACTION_CHIP, 201, INT_MAX, INT_MIN};
    uint64_t rng = f->state.rng_n;
    int actions = f->state.actions.stream.n_entries;
    CHECK(mask_one(f, chip));
    CHECK(!mask_one(f, extreme));
    CHECK(f->state.rng_n == rng && f->state.actions.stream.n_entries == actions);
    lw_entity_add_item_use(&f->actor, 201);
    CHECK(!mask_one(f, chip));
    free_fixture(f);

    f = make_fixture();
    lw_state_add_cooldown(&f->state, &f->actor, &f->chip, 2);
    CHECK(!mask_one(f, chip));
    free_fixture(f);
}

static void test_summon(void)
{
    fixture *f = make_fixture();
    LwEffectParameters effect;
    LwChip chip;
    LwBulbTemplate bulb;
    leekrl_action summon = {LEEKRL_ACTION_CHIP, 203, 1, 0};
    uint8_t mask;
    int result;
    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_SUMMON, 9000, 0, 0, 0, 0);
    lw_chip_init(&chip, 203, 2, 1, 6, &effect, 1, LW_ATTACK_LAUNCH_TYPE_CIRCLE,
                 1, 0, 0, 0, 0, 1, 203, "summon", LW_CHIP_TYPE_BULB, -1);
    lw_entity_add_chip(&f->actor, &chip);
    CHECK(leekrl_action_mask(&f->fight, &f->actor, &summon, 1, &mask) ==
          LEEKRL_ACTION_ENGINE_ERROR);
    lw_bulb_template_init(&bulb, 9000, "test", 100, 100, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 3, 3);
    CHECK(lw_bulbs_add_invocation_template(&bulb));
    CHECK(mask_one(f, summon));
    CHECK(leekrl_action_execute(&f->fight, &f->actor, &summon, &result) ==
          LEEKRL_ACTION_OK);
    CHECK(result > 0 && f->state.n_entities == 3);
    CHECK(!mask_one(f, summon));
    free_fixture(f);
    lw_bulbs_clear();
    lw_bulb_template_dispose(&bulb);
}

typedef struct {
    int observed;
    int moved;
    int fired;
    int chipped;
    int ended;
} live_result;

static int live_dispatch(LwFight *fight, LwEntity *actor, void *ai_file,
                         int turn, void *userdata)
{
    live_result *out = userdata;
    leekrl_action movement[] = {
        {LEEKRL_ACTION_MOVE, 0, 1, 0}, {LEEKRL_ACTION_MOVE, 0, -1, 0},
        {LEEKRL_ACTION_MOVE, 0, 0, 1}, {LEEKRL_ACTION_MOVE, 0, 0, -1}
    };
    leekrl_action casts[24], chip = {LEEKRL_ACTION_CHIP, 201, 0, 0};
    leekrl_action end = {LEEKRL_ACTION_END_TURN, 0, 0, 0};
    uint8_t mask[24];
    LwWeapon *weapon;
    size_t i, count;
    int result;
    (void)ai_file;
    (void)turn;
    if (lw_entity_get_team(actor) != 0) return 0;
    ++out->observed;
    CHECK(leekrl_action_mask(fight, actor, movement, 4, mask) == LEEKRL_ACTION_OK);
    for (i = 0; i < 4; ++i) if (mask[i]) {
        CHECK(leekrl_action_execute(fight, actor, &movement[i], &result) == LEEKRL_ACTION_OK);
        out->moved = 1;
        break;
    }
    weapon = lw_entity_get_weapon_at(actor, 0);
    CHECK(weapon);
    CHECK(leekrl_action_cast_layout(lw_weapon_get_attack(weapon), LEEKRL_ACTION_WEAPON,
                                    101, casts, 24, &count) == LEEKRL_ACTION_OK);
    CHECK(count == 24);
    CHECK(leekrl_action_mask(fight, actor, casts, count, mask) == LEEKRL_ACTION_OK);
    for (i = 0; i < count; ++i) if (mask[i]) {
        CHECK(leekrl_action_execute(fight, actor, &casts[i], &result) == LEEKRL_ACTION_OK);
        out->fired = 1;
        break;
    }
    CHECK(leekrl_action_mask(fight, actor, &chip, 1, mask) == LEEKRL_ACTION_OK && mask[0]);
    CHECK(leekrl_action_execute(fight, actor, &chip, &result) == LEEKRL_ACTION_OK);
    out->chipped = 1;
    CHECK(leekrl_action_execute(fight, actor, &end, &result) == LEEKRL_ACTION_OK);
    out->ended = 1;
    return 0;
}

static void test_live_fight(void)
{
    LwScenario *scenario = calloc(1, sizeof(*scenario));
    LwState *state = calloc(1, sizeof(*state));
    LwFight *fight = calloc(1, sizeof(*fight));
    LwOutcome *outcome = calloc(1, sizeof(*outcome));
    LwGenerator generator;
    LwEntityInfo info;
    LwEffectParameters effect;
    LwWeapon weapon;
    LwChip chip;
    live_result result = {0, 0, 0, 0, 0};
    int team;
    CHECK(scenario && state && fight && outcome);
    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_DAMAGE, 1, 1, 0,
                              LW_TARGET_ENEMIES | LW_TARGET_NON_SUMMONS, 0);
    lw_weapon_init(&weapon, 101, 2, 1, 6, &effect, 1,
                   LW_ATTACK_LAUNCH_TYPE_LINE, 1, 1, 101, "line", NULL, 0, 1);
    lw_chip_init(&chip, 201, 2, 0, 6, &effect, 1,
                 LW_ATTACK_LAUNCH_TYPE_CIRCLE, 1, 1, 2, 0, 0,
                 1, 201, "chip", LW_CHIP_TYPE_DAMAGE, 1);
    CHECK(lw_weapons_add_weapon(&weapon) && lw_chips_add_chip(&chip));
    lw_scenario_init(scenario);
    scenario->seed = 123;
    scenario->max_turns = 1;
    for (team = 0; team < 2; ++team) {
        lw_entity_info_init(&info);
        info.id = team + 1;
        info.life = 100;
        info.tp = 10;
        info.mp = 3;
        info.ram = 10;
        info.level = 1;
        info.weapons[0] = 101;
        info.n_weapons = 1;
        info.chips[0] = 201;
        info.n_chips = 1;
        CHECK(lw_scenario_add_entity(scenario, team, &info));
    }
    lw_generator_init(&generator);
    lw_generator_set_ai_dispatch(&generator, live_dispatch, &result);
    CHECK(lw_generator_run_scenario(&generator, scenario, NULL, NULL, NULL,
                                    state, fight, outcome) == 0);
    CHECK(result.observed && result.moved && result.fired && result.chipped && result.ended);
    lw_state_dispose(state);
    lw_scenario_dispose(scenario);
    lw_weapons_clear();
    lw_chips_clear();
    free(outcome);
    free(fight);
    free(state);
    free(scenario);
}

static void test_catalog(void)
{
    LwEffectParameters effect;
    LwWeapon a, b;
    LwChip chip;
    leekrl_action_catalog *first = NULL, *second = NULL;
    const leekrl_action *actions, *other;
    size_t i, n;
    lw_effect_parameters_init(&effect, LW_EFFECT_TYPE_DAMAGE, 1, 1, 0,
                              LW_TARGET_ENEMIES | LW_TARGET_NON_SUMMONS, 0);
    lw_weapon_init(&a, 109, 2, 1, 6, &effect, 1, LW_ATTACK_LAUNCH_TYPE_LINE,
                   1, 0, 109, "line", NULL, 0, 1);
    lw_weapon_init(&b, 108, 2, 1, 1, &effect, 1, LW_ATTACK_LAUNCH_TYPE_LINE,
                   1, 0, 108, "broadsword", NULL, 0, 1);
    lw_chip_init(&chip, 85, 2, 0, 0, &effect, 1, LW_ATTACK_LAUNCH_TYPE_CIRCLE,
                 1, 0, 0, 0, 0, 1, 85, "devil_strike", LW_CHIP_TYPE_DAMAGE, 1);
    CHECK(lw_weapons_add_weapon(&a) && lw_weapons_add_weapon(&b));
    CHECK(lw_chips_add_chip(&chip));
    CHECK(leekrl_action_catalog_create(&first) == LEEKRL_ACTION_OK);
    actions = leekrl_action_catalog_actions(first);
    n = leekrl_action_catalog_count(first);
    CHECK(n == 1 + LEEKRL_MOVE_ACTION_COUNT + 4 + 24 + 1);
    CHECK(actions[0].kind == LEEKRL_ACTION_END_TURN);
    for (i = 1; i <= LEEKRL_MOVE_ACTION_COUNT; ++i) {
        int d = abs(actions[i].dx) + abs(actions[i].dy);
        size_t j;
        CHECK(actions[i].kind == LEEKRL_ACTION_MOVE && d >= 1 && d <= 20);
        for (j = 1; j < i; ++j)
            CHECK(actions[i].dx != actions[j].dx || actions[i].dy != actions[j].dy);
    }
    for (; i < 1 + LEEKRL_MOVE_ACTION_COUNT + 4; ++i)
        CHECK(actions[i].kind == LEEKRL_ACTION_WEAPON && actions[i].item_id == 108);
    for (; i < n - 1; ++i)
        CHECK(actions[i].kind == LEEKRL_ACTION_WEAPON && actions[i].item_id == 109);
    CHECK(actions[i].kind == LEEKRL_ACTION_CHIP && actions[i].item_id == 85 &&
          actions[i].dx == 0 && actions[i].dy == 0);
    lw_weapons_clear();
    lw_chips_clear();
    CHECK(lw_chips_add_chip(&chip));
    CHECK(lw_weapons_add_weapon(&b) && lw_weapons_add_weapon(&a));
    CHECK(leekrl_action_catalog_create(&second) == LEEKRL_ACTION_OK);
    CHECK(leekrl_action_catalog_count(second) == n);
    other = leekrl_action_catalog_actions(second);
    for (i = 0; i < n; ++i)
        CHECK(actions[i].kind == other[i].kind && actions[i].item_id == other[i].item_id &&
              actions[i].dx == other[i].dx && actions[i].dy == other[i].dy);
    leekrl_action_catalog_destroy(first);
    leekrl_action_catalog_destroy(second);
    lw_weapons_clear();
    lw_chips_clear();
}

int main(void)
{
    test_line_mask_against_engine();
    test_chip_mask_against_engine();
    test_masks_change_with_state();
    test_movement_and_argument_errors();
    test_long_movement();
    test_obstacles_and_states();
    test_chip_limits();
    test_summon();
    test_live_fight();
    test_catalog();
    return 0;
}
