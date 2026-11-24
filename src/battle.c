#include "leekrl/battle.h"
#include "leekrl/observation.h"
#include "leekrl/plant_ai.h"
#include "leekrl/rules.h"

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
    leekrl_observer *observer;
    leekrl_battle_options options;
    size_t count;
    leekrl_reward_tracker rewards;
    float parts[LEEKRL_REWARD_PARTS];
    uint8_t map[LEEKRL_OBS_MAP_CELLS];
    char error[128];
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
    leekrl_reward_options_init(&options->rewards);
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
    if (!out) return LEEKRL_ENV_INVALID_ARGUMENT;
    *out = NULL;
    if (!valid_info(self) || !valid_info(opponent) || self->id == opponent->id ||
        !options || options->max_turns < 1 || options->max_turns > LW_FIGHT_MAX_TURNS ||
        !options->max_decisions || !leekrl_reward_options_valid(&options->rewards))
        return LEEKRL_ENV_INVALID_ARGUMENT;
    battle = calloc(1, sizeof(*battle));
    if (!battle) return LEEKRL_ENV_OUT_OF_MEMORY;
    battle->options = *options;
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
    battle->observer = leekrl_observer_create(battle->catalog);
    if (!battle->observer) {
        leekrl_battle_destroy(battle);
        return LEEKRL_ENV_OUT_OF_MEMORY;
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
    leekrl_observer_destroy(battle->observer);
    leekrl_action_catalog_destroy(battle->catalog);
    free(battle->state);
    free(battle->fight);
    free(battle->outcome);
    free(battle);
}

static LwEntity *find_entity(LwState *state, int id)
{
    int i, count;
    LwEntity **entities = lw_state_get_entities(state, &count);
    for (i = 0; i < count; ++i)
        if (lw_entity_get_id(entities[i]) == id) return entities[i];
    return NULL;
}

size_t leekrl_battle_observation_size(const leekrl_battle *battle)
{
    return battle ? leekrl_observer_float_count(battle->observer) + LEEKRL_OBS_MAP_CELLS : 0;
}

const leekrl_observer *leekrl_battle_observer(const leekrl_battle *battle)
{
    return battle ? battle->observer : NULL;
}

const char *leekrl_battle_error(const leekrl_battle *battle)
{
    return battle ? battle->error : "";
}

const float *leekrl_battle_reward_parts(const leekrl_battle *battle)
{
    return battle ? battle->parts : NULL;
}

int leekrl_battle_life(const leekrl_battle *battle, int team)
{
    LwEntity *leek = battle && battle->started && (team == 0 || team == 1)
        ? find_entity(battle->state, battle->scenario->entities[team][0].id) : NULL;
    return leek ? lw_entity_get_life(leek) : 0;
}

const leekrl_action_catalog *leekrl_battle_catalog(const leekrl_battle *battle)
{
    return battle ? battle->catalog : NULL;
}

/* The generic environment only knows floats: the map follows as floats. */
static int observe(leekrl_battle *battle, LwEntity *actor, float *out)
{
    size_t i, n = leekrl_observer_float_count(battle->observer);
    if (!leekrl_observe(battle->observer, battle->fight, actor, out, battle->map,
                        battle->error, sizeof(battle->error))) return -1;
    for (i = 0; i < LEEKRL_OBS_MAP_CELLS; ++i) out[n + i] = (float)battle->map[i];
    return 0;
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
    /* Before the end of the fight removes the summons. */
    leekrl_reward_step(&battle->rewards, &battle->options.rewards, battle->fight, 0,
                       battle->parts);
    if (lw_fight_step_finish(battle->fight, battle->scenario->draw_check_life)) return -1;
    leekrl_reward_end(&battle->rewards, &battle->options.rewards, battle->fight, 0,
                      battle->parts);
    return 0;
}

/* Rewards of a finished fight are already in parts. */
static int output(leekrl_battle *battle, float *observation, uint8_t *mask, int *finished)
{
    LwEntity *actor = lw_fight_step_current(battle->fight);
    *finished = !actor;
    if (!actor) {
        memset(observation, 0, leekrl_battle_observation_size(battle) * sizeof(*observation));
        memset(mask, 0, leekrl_action_catalog_count(battle->catalog));
        return 0;
    }
    leekrl_reward_step(&battle->rewards, &battle->options.rewards, battle->fight, 0,
                       battle->parts);
    return observe(battle, actor, observation) ||
           leekrl_action_catalog_mask(battle->catalog, battle->fight, actor, mask) !=
               LEEKRL_ACTION_OK;
}

static int reset_cb(void *context, uint64_t seed, float *observation, uint8_t *mask)
{
    leekrl_battle *battle = context;
    int finished;
    end_fight(battle);
    battle->error[0] = 0;
    battle->scenario->seed = (int)(seed & UINT64_C(0x7fffffff));
    battle->count = 0;
    if (lw_generator_prepare_scenario(&battle->generator, battle->scenario, NULL, NULL,
                                      NULL, battle->state, battle->fight,
                                      battle->outcome)) return -1;
    battle->started = 1;
    if (lw_fight_step_start(battle->fight)) return -1;
    leekrl_reward_start(&battle->rewards, &battle->options.rewards, battle->fight);
    /* What happens before the first decision rewards no action. */
    return advance(battle) || output(battle, observation, mask, &finished) || finished;
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
    if (chosen->kind != LEEKRL_ACTION_END_TURN) leekrl_reward_count_action(&battle->rewards, 0);
    if (chosen->kind == LEEKRL_ACTION_END_TURN || lw_entity_is_dead(actor) ||
        lw_state_is_finished(battle->state)) {
        if (lw_fight_step_end_turn(battle->fight) || advance(battle)) return -1;
    }
    if (output(battle, observation, mask, &finished)) return -1;
    *reward = leekrl_reward_sum(battle->parts);
    *terminated = finished;
    *truncated = !finished && battle->count == battle->options.max_decisions;
    return 0;
}

leekrl_env_callbacks leekrl_battle_callbacks(void)
{
    leekrl_env_callbacks callbacks = {reset_cb, step_cb};
    return callbacks;
}
