#include "leekrl/battle.h"
#include "leekrl/grid.h"
#include "leekrl/plant_ai.h"
#include "leekrl/rules.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lw_chip.h"
#include "lw_entity.h"
#include "lw_generator.h"
#include "lw_state.h"
#include "lw_weapon.h"

struct leekrl_battle {
    LwScenario *scenario;
    leekrl_action_catalog *catalog;
    leekrl_battle_options options;
    size_t count;
    size_t observation_size;
    LwGenerator generator;
    LwState *state;
    LwFight *fight;
    LwOutcome *outcome;
    int started;
};

void leekrl_battle_options_init(leekrl_battle_options *options)
{
    if (!options) return;
    options->max_turns = LW_FIGHT_MAX_TURNS;
    options->max_decisions = 4096;
    options->grid_radius = 17;
    options->win_reward = 1;
    options->loss_reward = -1;
    options->draw_reward = 0;
    options->step_reward = 0;
}

static int valid_info(const LwEntityInfo *info)
{
    int i, j;
    if (!info || info->type != LW_ENTITY_TYPE_LEEK || info->dead || info->life <= 0 ||
        info->n_weapons < 0 || info->n_weapons > 4 ||
        info->n_chips < 0 || info->n_chips > LW_ENTITY_INFO_MAX_CHIPS ||
        info->n_start_effects < 0 ||
        (info->n_start_effects && !info->start_effects)) return 0;
    for (i = 0; i < info->n_weapons; ++i) {
        if (!lw_weapons_get_weapon(info->weapons[i]) ||
            leekrl_solo_item_excluded(info->weapons[i])) return 0;
        for (j = 0; j < i; ++j)
            if (info->weapons[i] == info->weapons[j]) return 0;
    }
    for (i = 0; i < info->n_chips; ++i) {
        if (!lw_chips_get_chip(info->chips[i]) ||
            leekrl_solo_item_excluded(info->chips[i])) return 0;
        for (j = 0; j < i; ++j)
            if (info->chips[i] == info->chips[j]) return 0;
    }
    return 1;
}

leekrl_env_status leekrl_battle_create(const LwEntityInfo *self,
                                       const LwEntityInfo *opponent,
                                       const leekrl_battle_options *options,
                                       leekrl_battle **out)
{
    leekrl_battle *battle;
    leekrl_action_status status;
    size_t grid_size;
    leekrl_grid_layout layout;
    if (!out) return LEEKRL_ENV_INVALID_ARGUMENT;
    *out = NULL;
    if (!valid_info(self) || !valid_info(opponent) || self->id == opponent->id ||
        !options || options->max_turns < 1 || options->max_turns > LW_FIGHT_MAX_TURNS ||
        !options->max_decisions ||
        !isfinite(options->win_reward) || !isfinite(options->loss_reward) ||
        !isfinite(options->draw_reward) || !isfinite(options->step_reward))
        return LEEKRL_ENV_INVALID_ARGUMENT;
    if (leekrl_grid_layout_for(options->grid_radius, &layout) != LEEKRL_GRID_OK ||
        layout.width > SIZE_MAX / layout.height ||
        layout.width * layout.height > SIZE_MAX / layout.channels)
        return LEEKRL_ENV_INVALID_ARGUMENT;
    grid_size = layout.width * layout.height * layout.channels;
    if (grid_size > SIZE_MAX - LEEKRL_BATTLE_GRID) return LEEKRL_ENV_INVALID_ARGUMENT;
    battle = calloc(1, sizeof(*battle));
    if (!battle) return LEEKRL_ENV_OUT_OF_MEMORY;
    battle->options = *options;
    battle->observation_size = LEEKRL_BATTLE_GRID + grid_size;
    battle->scenario = calloc(1, sizeof(*battle->scenario));
    battle->state = calloc(1, sizeof(*battle->state));
    battle->fight = calloc(1, sizeof(*battle->fight));
    battle->outcome = calloc(1, sizeof(*battle->outcome));
    if (!battle->scenario || !battle->state || !battle->fight || !battle->outcome) {
        leekrl_battle_destroy(battle);
        return LEEKRL_ENV_OUT_OF_MEMORY;
    }
    lw_generator_init(&battle->generator);
    lw_generator_set_plant_ai_dispatch(&battle->generator, leekrl_plant_ai, NULL);
    lw_scenario_init(battle->scenario);
    battle->scenario->max_turns = options->max_turns;
    if (!lw_scenario_add_entity(battle->scenario, 0, self) ||
        !lw_scenario_add_entity(battle->scenario, 1, opponent)) {
        leekrl_battle_destroy(battle);
        return LEEKRL_ENV_OUT_OF_MEMORY;
    }
    status = leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
                                                  leekrl_solo_excluded_count,
                                                  &battle->catalog);
    if (status != LEEKRL_ACTION_OK) {
        leekrl_battle_destroy(battle);
        return status == LEEKRL_ACTION_OUT_OF_MEMORY ? LEEKRL_ENV_OUT_OF_MEMORY :
               LEEKRL_ENV_INVALID_ARGUMENT;
    }
    *out = battle;
    return LEEKRL_ENV_OK;
}

static void end_fight(leekrl_battle *battle)
{
    if (battle->started) lw_state_dispose(battle->state);
    battle->started = 0;
}

void leekrl_battle_destroy(leekrl_battle *battle)
{
    if (!battle) return;
    end_fight(battle);
    if (battle->scenario) {
        lw_scenario_dispose(battle->scenario);
        free(battle->scenario);
    }
    leekrl_action_catalog_destroy(battle->catalog);
    free(battle->state);
    free(battle->fight);
    free(battle->outcome);
    free(battle);
}

size_t leekrl_battle_observation_size(const leekrl_battle *battle)
{
    return battle ? battle->observation_size : 0;
}

const leekrl_action_catalog *leekrl_battle_catalog(const leekrl_battle *battle)
{
    return battle ? battle->catalog : NULL;
}

static LwEntity *find_entity(LwState *state, int id)
{
    int i, count;
    LwEntity **entities = lw_state_get_entities(state, &count);
    for (i = 0; i < count; ++i)
        if (lw_entity_get_id(entities[i]) == id) return entities[i];
    return NULL;
}

static int observe(leekrl_battle *battle, LwEntity *actor, float *o)
{
    LwState *state = battle->state;
    LwEntity *enemy = find_entity(state, battle->scenario->entities[1][0].id);
    size_t required;
    o[LEEKRL_BATTLE_TURN] = (float)lw_state_get_turn(state);
    o[LEEKRL_BATTLE_SELF_LIFE] = actor ? (float)lw_entity_get_life(actor) : 0;
    o[LEEKRL_BATTLE_ENEMY_LIFE] = enemy ? (float)lw_entity_get_life(enemy) : 0;
    o[LEEKRL_BATTLE_TP] = actor ? (float)lw_entity_get_tp(actor) : 0;
    o[LEEKRL_BATTLE_MP] = actor ? (float)lw_entity_get_mp(actor) : 0;
    if (!actor || !lw_entity_get_cell(actor)) {
        memset(o + LEEKRL_BATTLE_GRID, 0,
               (battle->observation_size - LEEKRL_BATTLE_GRID) * sizeof(*o));
        return 0;
    }
    return leekrl_grid_observation(battle->fight, actor, battle->options.grid_radius,
                                   o + LEEKRL_BATTLE_GRID,
                                   battle->observation_size - LEEKRL_BATTLE_GRID,
                                   &required) != LEEKRL_GRID_OK;
}

/* Play turns until the learner's team has a decision to make, or the fight is
 * over. The opponent's team only passes. */
static int advance(leekrl_battle *battle)
{
    LwEntity *actor;
    while (!lw_fight_step_is_over(battle->fight)) {
        if (lw_fight_step_begin_turn(battle->fight, &actor)) return -1;
        if (!actor) continue;
        if (lw_entity_get_team(actor) == 0) return 0;
        if (lw_fight_step_end_turn(battle->fight)) return -1;
    }
    return lw_fight_step_finish(battle->fight, battle->scenario->draw_check_life);
}

static int output(leekrl_battle *battle, float *observation, uint8_t *mask,
                  int *finished, float *reward)
{
    LwEntity *actor = lw_fight_step_current(battle->fight);
    size_t count = leekrl_action_catalog_count(battle->catalog);
    int winner;
    *finished = !actor;
    *reward = 0;
    if (actor) {
        return observe(battle, actor, observation) ||
               leekrl_action_catalog_mask(battle->catalog, battle->fight, actor,
                                          mask) != LEEKRL_ACTION_OK;
    }
    actor = find_entity(battle->state, battle->scenario->entities[0][0].id);
    if (observe(battle, actor, observation)) return -1;
    memset(mask, 0, count);
    winner = lw_fight_get_winner(battle->fight);
    if (winner == 0) *reward = battle->options.win_reward;
    else if (winner == 1) *reward = battle->options.loss_reward;
    else *reward = battle->options.draw_reward;
    return 0;
}

static int reset_cb(void *context, uint64_t seed, float *observation, uint8_t *mask)
{
    leekrl_battle *battle = context;
    int finished;
    float reward;
    end_fight(battle);
    battle->scenario->seed = (int)(seed & UINT64_C(0x7fffffff));
    battle->count = 0;
    if (lw_generator_prepare_scenario(&battle->generator, battle->scenario, NULL, NULL,
                                      NULL, battle->state, battle->fight,
                                      battle->outcome)) return -1;
    battle->started = 1;
    return lw_fight_step_start(battle->fight) || advance(battle) ||
           output(battle, observation, mask, &finished, &reward) || finished;
}

static int step_cb(void *context, size_t action, float *observation, uint8_t *mask,
                   float *reward, int *terminated, int *truncated)
{
    leekrl_battle *battle = context;
    const leekrl_action *chosen;
    LwEntity *actor = lw_fight_step_current(battle->fight);
    int finished, engine_result;
    if (!battle->started || !actor || battle->count >= battle->options.max_decisions ||
        action >= leekrl_action_catalog_count(battle->catalog)) return -1;
    chosen = leekrl_action_catalog_actions(battle->catalog) + action;
    if (leekrl_action_execute(battle->fight, actor, chosen, &engine_result) !=
        LEEKRL_ACTION_OK) return -1;
    ++battle->count;
    if (chosen->kind == LEEKRL_ACTION_END_TURN || lw_entity_is_dead(actor) ||
        lw_state_is_finished(battle->state)) {
        if (lw_fight_step_end_turn(battle->fight) || advance(battle)) return -1;
    }
    if (output(battle, observation, mask, &finished, reward)) return -1;
    *reward += battle->options.step_reward;
    *terminated = finished;
    *truncated = !finished && battle->count == battle->options.max_decisions;
    return 0;
}

leekrl_env_callbacks leekrl_battle_callbacks(void)
{
    leekrl_env_callbacks callbacks = {reset_cb, step_cb};
    return callbacks;
}
