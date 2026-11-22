#include "leekrl/battle.h"
#include "leekrl/grid.h"

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
    size_t *history;
    size_t count;
    size_t capacity;
    size_t observation_size;
};

typedef struct {
    leekrl_battle *battle;
    size_t seen;
    int captured;
    int failed;
    float *observation;
    uint8_t *mask;
} battle_replay;

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
        if (!lw_weapons_get_weapon(info->weapons[i])) return 0;
        for (j = 0; j < i; ++j)
            if (info->weapons[i] == info->weapons[j]) return 0;
    }
    for (i = 0; i < info->n_chips; ++i) {
        if (!lw_chips_get_chip(info->chips[i])) return 0;
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
        !options->max_decisions || options->max_decisions > SIZE_MAX / sizeof(size_t) ||
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
    if (!battle->scenario) {
        leekrl_battle_destroy(battle);
        return LEEKRL_ENV_OUT_OF_MEMORY;
    }
    lw_scenario_init(battle->scenario);
    battle->scenario->max_turns = options->max_turns;
    if (!lw_scenario_add_entity(battle->scenario, 0, self) ||
        !lw_scenario_add_entity(battle->scenario, 1, opponent)) {
        leekrl_battle_destroy(battle);
        return LEEKRL_ENV_OUT_OF_MEMORY;
    }
    status = leekrl_action_catalog_create(&battle->catalog);
    if (status != LEEKRL_ACTION_OK) {
        leekrl_battle_destroy(battle);
        return status == LEEKRL_ACTION_OUT_OF_MEMORY ? LEEKRL_ENV_OUT_OF_MEMORY :
               LEEKRL_ENV_INVALID_ARGUMENT;
    }
    *out = battle;
    return LEEKRL_ENV_OK;
}

void leekrl_battle_destroy(leekrl_battle *battle)
{
    if (!battle) return;
    if (battle->scenario) {
        lw_scenario_dispose(battle->scenario);
        free(battle->scenario);
    }
    leekrl_action_catalog_destroy(battle->catalog);
    free(battle->history);
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

static int observe(battle_replay *replay, LwFight *fight, LwEntity *actor)
{
    LwState *state = lw_fight_get_state(fight);
    LwEntity *enemy = find_entity(state, replay->battle->scenario->entities[1][0].id);
    float *o = replay->observation;
    size_t required;
    o[LEEKRL_BATTLE_TURN] = (float)lw_state_get_turn(state);
    o[LEEKRL_BATTLE_SELF_LIFE] = actor ? (float)lw_entity_get_life(actor) : 0;
    o[LEEKRL_BATTLE_ENEMY_LIFE] = enemy ? (float)lw_entity_get_life(enemy) : 0;
    o[LEEKRL_BATTLE_TP] = actor ? (float)lw_entity_get_tp(actor) : 0;
    o[LEEKRL_BATTLE_MP] = actor ? (float)lw_entity_get_mp(actor) : 0;
    if (!actor || !lw_entity_get_cell(actor)) {
        memset(o + LEEKRL_BATTLE_GRID, 0,
               (replay->battle->observation_size - LEEKRL_BATTLE_GRID) * sizeof(*o));
        return 0;
    }
    return leekrl_grid_observation(fight, actor, replay->battle->options.grid_radius,
                                   o + LEEKRL_BATTLE_GRID,
                                   replay->battle->observation_size - LEEKRL_BATTLE_GRID,
                                   &required) != LEEKRL_GRID_OK;
}

static int dispatch(LwFight *fight, LwEntity *actor, void *ai_file,
                    int turn, void *userdata)
{
    battle_replay *r = userdata;
    LwState *state = lw_fight_get_state(fight);
    const leekrl_action *actions = leekrl_action_catalog_actions(r->battle->catalog);
    size_t action_count = leekrl_action_catalog_count(r->battle->catalog);
    (void)ai_file;
    (void)turn;
    if (r->captured || r->failed ||
        lw_entity_get_team(actor) != 0 ||
        lw_entity_get_id(actor) != r->battle->scenario->entities[0][0].id)
        return 0;
    while (!lw_state_is_finished(state)) {
        size_t action;
        int engine_result;
        if (r->seen == r->battle->count) {
            if (observe(r, fight, actor) ||
                leekrl_action_mask(fight, actor, actions, action_count, r->mask) !=
                    LEEKRL_ACTION_OK)
                r->failed = 1;
            else
                r->captured = 1;
            return 0;
        }
        action = r->battle->history[r->seen++];
        if (action >= action_count ||
            leekrl_action_execute(fight, actor, &actions[action], &engine_result) !=
                LEEKRL_ACTION_OK) {
            r->failed = 1;
            return 0;
        }
        if (actions[action].kind == LEEKRL_ACTION_END_TURN) return 0;
    }
    return 0;
}

static int run(leekrl_battle *battle, float *observation, uint8_t *mask,
               int *finished, float *reward)
{
    LwState *state = calloc(1, sizeof(*state));
    LwFight *fight = calloc(1, sizeof(*fight));
    LwOutcome *outcome = calloc(1, sizeof(*outcome));
    LwGenerator generator;
    battle_replay replay;
    int rc = -1;
    if (!state || !fight || !outcome) goto done;
    memset(&replay, 0, sizeof(replay));
    replay.battle = battle;
    replay.observation = observation;
    replay.mask = mask;
    lw_generator_init(&generator);
    lw_generator_set_ai_dispatch(&generator, dispatch, &replay);
    if (lw_generator_run_scenario(&generator, battle->scenario, NULL, NULL, NULL,
                                  state, fight, outcome) != 0 ||
        replay.failed || replay.seen != battle->count) goto dispose;
    *finished = !replay.captured;
    *reward = 0;
    if (*finished) {
        LwEntity *actor = find_entity(state, battle->scenario->entities[0][0].id);
        if (observe(&replay, fight, actor)) goto dispose;
        memset(mask, 0, leekrl_action_catalog_count(battle->catalog));
        if (outcome->winner == 0) *reward = battle->options.win_reward;
        else if (outcome->winner == 1) *reward = battle->options.loss_reward;
        else *reward = battle->options.draw_reward;
    }
    rc = 0;
dispose:
    lw_state_dispose(state);
done:
    free(outcome);
    free(fight);
    free(state);
    return rc;
}

static int reset_cb(void *context, uint64_t seed, float *observation, uint8_t *mask)
{
    leekrl_battle *battle = context;
    int finished;
    float reward;
    battle->scenario->seed = (int)(seed & UINT64_C(0x7fffffff));
    battle->count = 0;
    return run(battle, observation, mask, &finished, &reward) || finished;
}

static int step_cb(void *context, size_t action, float *observation, uint8_t *mask,
                   float *reward, int *terminated, int *truncated)
{
    leekrl_battle *battle = context;
    int finished;
    size_t capacity;
    size_t *history;
    if (battle->count >= battle->options.max_decisions ||
        action >= leekrl_action_catalog_count(battle->catalog)) return -1;
    if (battle->count == battle->capacity) {
        capacity = battle->capacity ? battle->capacity : 16;
        if (capacity > battle->options.max_decisions / 2)
            capacity = battle->options.max_decisions;
        else if (battle->capacity) capacity *= 2;
        if (capacity > battle->options.max_decisions) capacity = battle->options.max_decisions;
        history = realloc(battle->history, capacity * sizeof(*history));
        if (!history) return -1;
        battle->history = history;
        battle->capacity = capacity;
    }
    battle->history[battle->count++] = action;
    if (run(battle, observation, mask, &finished, reward)) return -1;
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
