#include "leekrl/vec.h"

#include "leekrl/actions.h"
#include "leekrl/plant_ai.h"
#include "leekrl/rules.h"
#include "leekrl/threads.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lw_area.h"
#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_generator.h"
#include "lw_map.h"
#include "lw_outcome.h"
#include "lw_scenario.h"
#include "lw_state.h"

#define MAX_REDRAWS 1000
#define MAX_RESTARTS 1000

typedef struct {
    leekrl_vec *vec;
    int index;
    uint64_t rng;
    LwGenerator generator;
    LwScenario scenario;
    LwState state;
    LwFight fight;
    LwOutcome outcome;
    int scenario_ready;
    int started;
    leekrl_reward_tracker rewards;
    uint8_t *legal;              /* mask, one byte per action */
    int decisions[2];
    int known[2];
    int redrawn;
    int start_life[2];
    int fight_seed;
    char error[192];
} environment;

struct leekrl_vec {
    leekrl_vec_options options;
    leekrl_vec_buffers buffers;
    leekrl_action_catalog *catalog;
    leekrl_observer *observer;
    leekrl_build_pool *pool;
    leekrl_threads *threads;
    environment **envs;
    size_t actions;
    size_t floats;
    size_t mask_bytes;
    leekrl_build *known;
    double *weights;
    double total_weight;
    int n_known;
    const int32_t *chosen;
    int ready;
    char error[256];
};

void leekrl_vec_options_init(leekrl_vec_options *options)
{
    if (!options) return;
    memset(options, 0, sizeof(*options));
    options->envs = 1;
    options->threads = 1;
    options->known_probability = 0.01;
    options->skip_forced = 1;
    options->trim = LEEKRL_TRIM_HARMFUL;
    leekrl_build_options_init(&options->builds);
    leekrl_reward_options_init(&options->rewards);
}

static uint64_t draw(uint64_t *rng)
{
    uint64_t z = (*rng += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static double draw_unit(uint64_t *rng)
{
    return (double)(draw(rng) >> 11) / 9007199254740992.0;
}

static int fail(environment *env, const char *message, const char *detail)
{
    snprintf(env->error, sizeof(env->error), "environment %d, fight seed %d, turn %d: %s%s%s",
             env->index, env->fight_seed,
             env->started ? lw_fight_get_turn(&env->fight) : 0, message,
             detail && *detail ? ": " : "", detail ? detail : "");
    return 0;
}

/* ---- Fights ----------------------------------------------------------- */

static void stop_fight(environment *env)
{
    if (env->started) lw_state_dispose(&env->state);
    if (env->scenario_ready) lw_scenario_dispose(&env->scenario);
    env->started = env->scenario_ready = 0;
}

static int choose_build(environment *env, int team, leekrl_build *build)
{
    leekrl_vec *vec = env->vec;
    env->known[team] = 0;
    if (vec->n_known && draw_unit(&env->rng) < vec->options.known_probability) {
        double pick = draw_unit(&env->rng) * vec->total_weight;
        int i = 0;
        while (i < vec->n_known - 1 && pick >= vec->weights[i]) pick -= vec->weights[i++];
        *build = vec->known[i];
        env->known[team] = i + 1;
        return 1;
    }
    return leekrl_build_random(vec->pool, &vec->options.builds, &env->rng, build);
}

static LwEntity *leek_of(environment *env, int team)
{
    int i;
    for (i = 0; i < env->state.n_entities; ++i) {
        LwEntity *e = env->state.m_entities[i];
        if (!lw_entity_is_summon(e) && lw_entity_get_team(e) == team) return e;
    }
    return NULL;
}

static int start_fight(environment *env)
{
    leekrl_build builds[2];
    int team;
    for (team = 0; team < 2; ++team)
        if (!choose_build(env, team, &builds[team])) return fail(env, "no random build", NULL);
    env->redrawn = 0;
    for (;;) {
        LwEntity *a, *b;
        stop_fight(env);
        env->fight_seed = (int)(draw(&env->rng) & 0x7fffffff);
        lw_scenario_init(&env->scenario);
        env->scenario_ready = 1;
        env->scenario.seed = env->fight_seed;
        for (team = 0; team < 2; ++team) {
            LwEntityInfo info;
            leekrl_build_to_entity(&builds[team], team + 1, &info);
            if (!lw_scenario_add_entity(&env->scenario, team, &info))
                return fail(env, "scenario refused a leek", NULL);
        }
        memset(&env->state, 0, sizeof(env->state));
        memset(&env->fight, 0, sizeof(env->fight));
        if (lw_generator_prepare_scenario(&env->generator, &env->scenario, NULL, NULL, NULL,
                                          &env->state, &env->fight, &env->outcome))
            return fail(env, "scenario refused", NULL);
        env->started = 1;
        env->state.actions.stream.enabled = 0; /* nobody reads the report */
        if (lw_fight_step_start(&env->fight))
            return fail(env, "fight cannot start", lw_state_get_fatal_error_message(&env->state));
        a = leek_of(env, 0);
        b = leek_of(env, 1);
        if (!a || !b || !lw_entity_get_cell(a) || !lw_entity_get_cell(b))
            return fail(env, "missing leek", NULL);
        /* The generator keeps its last try when it finds no connected map. */
        if (lw_entity_get_cell(a)->composante == lw_entity_get_cell(b)->composante) break;
        if (++env->redrawn > MAX_REDRAWS) return fail(env, "no connected map", NULL);
    }
    for (team = 0; team < 2; ++team) {
        env->decisions[team] = 0;
        env->start_life[team] = lw_entity_get_total_life(leek_of(env, team));
    }
    leekrl_reward_start(&env->rewards, &env->vec->options.rewards, &env->fight);
    return 1;
}

static int finish_fight(environment *env)
{
    leekrl_vec *vec = env->vec;
    float *parts = vec->buffers.end_rewards + 2 * LEEKRL_REWARD_PARTS * (size_t)env->index;
    float *episode = vec->buffers.episode + LEEKRL_EPISODE_SIZE * (size_t)env->index;
    int team;
    for (team = 0; team < 2; ++team) {
        LwEntity *leek = leek_of(env, team);
        leekrl_reward_step(&env->rewards, &vec->options.rewards, &env->fight, team,
                           parts + team * LEEKRL_REWARD_PARTS);
        episode[LEEKRL_EPISODE_LIFE + team] =
            leek && env->start_life[team] > 0
                ? (float)lw_entity_get_life(leek) / (float)env->start_life[team] : 0;
        episode[LEEKRL_EPISODE_DECISIONS + team] = (float)env->decisions[team];
        episode[LEEKRL_EPISODE_ACTIONS + team] = (float)env->rewards.actions[team];
        episode[LEEKRL_EPISODE_KNOWN + team] = (float)env->known[team];
    }
    if (lw_fight_step_finish(&env->fight, 0))
        return fail(env, "fight cannot finish", lw_state_get_fatal_error_message(&env->state));
    for (team = 0; team < 2; ++team)
        leekrl_reward_end(&env->rewards, &vec->options.rewards, &env->fight, team,
                          parts + team * LEEKRL_REWARD_PARTS);
    episode[LEEKRL_EPISODE_TURNS] = (float)lw_fight_get_turn(&env->fight);
    episode[LEEKRL_EPISODE_WINNER] = (float)lw_fight_get_winner(&env->fight);
    episode[LEEKRL_EPISODE_REDRAWN] = (float)env->redrawn;
    vec->buffers.done[env->index] = 1;
    return 1;
}

/* ---- Decisions -------------------------------------------------------- */

static int only_end_turn(const uint8_t *legal, size_t count)
{
    return !memchr(legal + 1, 1, count - 1);
}

/* Write everything the learner needs for the decision of actor. */
static int write_decision(environment *env, LwEntity *actor)
{
    leekrl_vec *vec = env->vec;
    const leekrl_vec_buffers *out = &vec->buffers;
    size_t index = (size_t)env->index, i;
    uint8_t *mask = out->mask + index * vec->mask_bytes;
    float *parts = out->rewards + index * LEEKRL_REWARD_PARTS;
    int team = lw_entity_get_team(actor);
    char detail[128];
    if (!leekrl_observe(vec->observer, &env->fight, actor, out->floats + index * vec->floats,
                        out->map + index * LEEKRL_OBS_MAP_CELLS, detail, sizeof(detail)))
        return fail(env, "cannot observe", detail);
    memset(mask, 0, vec->mask_bytes);
    for (i = 0; i < vec->actions; ++i) mask[i >> 3] |= (uint8_t)(env->legal[i] << (i & 7));
    leekrl_reward_step(&env->rewards, &vec->options.rewards, &env->fight, team, parts);
    out->team[index] = (uint8_t)team;
    out->first[index] = env->decisions[team] == 0;
    /* What happened before its first decision rewards no decision. */
    if (out->first[index]) memset(parts, 0, LEEKRL_REWARD_PARTS * sizeof(*parts));
    ++env->decisions[team];
    return 1;
}

/* Play until a decision is needed, starting new fights as needed. */
static int advance(environment *env)
{
    leekrl_vec *vec = env->vec;
    int restarts = 0;
    for (;;) {
        LwEntity *actor = lw_fight_step_current(&env->fight);
        if (actor) {
            if (!lw_entity_is_dead(actor) && !lw_state_is_finished(&env->state)) {
                leekrl_action_status status = leekrl_action_catalog_trimmed_mask(
                    vec->catalog, &env->fight, actor, vec->options.trim, env->legal);
                if (status != LEEKRL_ACTION_OK)
                    return fail(env, "cannot compute the mask",
                                leekrl_action_status_string(status));
                if (!vec->options.skip_forced || !only_end_turn(env->legal, vec->actions))
                    return write_decision(env, actor);
            }
            if (lw_fight_step_end_turn(&env->fight))
                return fail(env, "cannot end the turn",
                            lw_state_get_fatal_error_message(&env->state));
        } else if (lw_fight_step_is_over(&env->fight)) {
            /* A fight where nobody could decide anything is only replaced. */
            if ((env->decisions[0] || env->decisions[1]) && !finish_fight(env)) return 0;
            if (++restarts > MAX_RESTARTS) return fail(env, "fights without any decision", NULL);
            if (!start_fight(env)) return 0;
        } else if (lw_fight_step_begin_turn(&env->fight, &actor)) {
            return fail(env, "cannot begin a turn",
                        lw_state_get_fatal_error_message(&env->state));
        }
    }
}

static void reset_task(void *context, int index)
{
    leekrl_vec *vec = context;
    environment *env = vec->envs[index];
    uint64_t seed = vec->options.seed;
    stop_fight(env);
    env->error[0] = 0;
    vec->buffers.done[index] = 0;
    /* Unrelated streams for each environment. */
    env->rng = draw(&seed) ^ ((uint64_t)index + 1) * UINT64_C(0xD6E8FEB86659FD93);
    (void)draw(&env->rng);
    if (start_fight(env)) advance(env);
}

static void step_task(void *context, int index)
{
    leekrl_vec *vec = context;
    environment *env = vec->envs[index];
    int32_t chosen = vec->chosen[index];
    const leekrl_action *action;
    LwEntity *actor = lw_fight_step_current(&env->fight);
    leekrl_action_status status;
    int result;
    vec->buffers.done[index] = 0;
    if (chosen == LEEKRL_VEC_WAIT) return;
    if (!actor || chosen < 0 || (size_t)chosen >= vec->actions || !env->legal[chosen]) {
        char detail[32];
        snprintf(detail, sizeof(detail), "%d", (int)chosen);
        fail(env, "illegal action", detail);
        return;
    }
    action = leekrl_action_catalog_actions(vec->catalog) + chosen;
    status = leekrl_action_execute(&env->fight, actor, action, &result);
    if (status != LEEKRL_ACTION_OK) {
        fail(env, leekrl_action_status_string(status),
             lw_state_get_fatal_error_message(&env->state));
        return;
    }
    if (action->kind == LEEKRL_ACTION_END_TURN) {
        if (lw_fight_step_end_turn(&env->fight)) {
            fail(env, "cannot end the turn", lw_state_get_fatal_error_message(&env->state));
            return;
        }
    } else {
        leekrl_reward_count_action(&env->rewards, lw_entity_get_team(actor));
    }
    advance(env);
}

/* ---- Public functions -------------------------------------------------- */

static int run(leekrl_vec *vec, leekrl_task task)
{
    int i;
    leekrl_threads_run(vec->threads, task, vec, vec->options.envs);
    for (i = 0; i < vec->options.envs; ++i) {
        if (!vec->envs[i]->error[0]) continue;
        snprintf(vec->error, sizeof(vec->error), "%s", vec->envs[i]->error);
        vec->ready = 0;
        return 0;
    }
    vec->ready = 1;
    return 1;
}

int leekrl_vec_reset(leekrl_vec *vec)
{
    if (!vec) return 0;
    vec->error[0] = 0;
    return run(vec, reset_task);
}

int leekrl_vec_step(leekrl_vec *vec, const int32_t *actions)
{
    if (!vec || !actions) return 0;
    if (!vec->ready) {
        if (!vec->error[0]) snprintf(vec->error, sizeof(vec->error), "not reset");
        return 0;
    }
    vec->chosen = actions;
    return run(vec, step_task);
}

const char *leekrl_vec_error(const leekrl_vec *vec) { return vec ? vec->error : ""; }
size_t leekrl_vec_actions(const leekrl_vec *vec) { return vec ? vec->actions : 0; }
size_t leekrl_vec_floats(const leekrl_vec *vec) { return vec ? vec->floats : 0; }
size_t leekrl_vec_mask_bytes(const leekrl_vec *vec) { return vec ? vec->mask_bytes : 0; }

int leekrl_vec_sizes(size_t *actions, size_t *floats, size_t *mask_bytes)
{
    leekrl_action_catalog *catalog = NULL;
    leekrl_observer *observer;
    size_t count;
    if (!actions || !floats || !mask_bytes ||
        leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
                                             leekrl_solo_excluded_count, &catalog) !=
            LEEKRL_ACTION_OK) return 0;
    count = leekrl_action_catalog_count(catalog);
    observer = leekrl_observer_create(catalog);
    if (observer && leekrl_observer_weapons(observer)) {
        *actions = count;
        *floats = leekrl_observer_float_count(observer);
        *mask_bytes = (count + 7) / 8;
    } else {
        count = 0;
    }
    leekrl_observer_destroy(observer);
    leekrl_action_catalog_destroy(catalog);
    return count != 0;
}

int leekrl_vec_action_table(int32_t *table)
{
    leekrl_action_catalog *catalog = NULL;
    leekrl_observer *observer;
    const leekrl_action *actions;
    size_t i, n;
    if (!table || leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
                                                       leekrl_solo_excluded_count, &catalog) !=
                      LEEKRL_ACTION_OK) return 0;
    observer = leekrl_observer_create(catalog);
    actions = leekrl_action_catalog_actions(catalog);
    n = observer ? leekrl_action_catalog_count(catalog) : 0;
    for (i = 0; i < n; ++i) {
        const leekrl_action *a = &actions[i];
        table[4 * i] = a->kind;
        table[4 * i + 1] = a->kind == LEEKRL_ACTION_WEAPON
            ? leekrl_observer_weapon_index(observer, a->item_id)
            : a->kind == LEEKRL_ACTION_CHIP ? leekrl_observer_chip_index(observer, a->item_id) : -1;
        table[4 * i + 2] = a->dx;
        table[4 * i + 3] = a->dy;
    }
    leekrl_observer_destroy(observer);
    leekrl_action_catalog_destroy(catalog);
    return n != 0;
}

int leekrl_vec_items(int32_t *weapons, int32_t *chips)
{
    leekrl_action_catalog *catalog = NULL;
    leekrl_observer *observer;
    const leekrl_action *actions;
    size_t i, n;
    if (!weapons || !chips ||
        leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
                                             leekrl_solo_excluded_count, &catalog) !=
            LEEKRL_ACTION_OK) return 0;
    observer = leekrl_observer_create(catalog);
    actions = leekrl_action_catalog_actions(catalog);
    n = observer ? leekrl_action_catalog_count(catalog) : 0;
    for (i = 0; i < n; ++i) {
        int id = actions[i].item_id;
        if (actions[i].kind == LEEKRL_ACTION_WEAPON)
            weapons[leekrl_observer_weapon_index(observer, id)] = id;
        else if (actions[i].kind == LEEKRL_ACTION_CHIP)
            chips[leekrl_observer_chip_index(observer, id)] = id;
    }
    leekrl_observer_destroy(observer);
    leekrl_action_catalog_destroy(catalog);
    return n != 0;
}

void leekrl_vec_layout(leekrl_vec_layout_info *layout)
{
    if (!layout) return;
    layout->map_cells = LEEKRL_OBS_MAP_CELLS;
    layout->reward_parts = LEEKRL_REWARD_PARTS;
    layout->episode_size = LEEKRL_EPISODE_SIZE;
    layout->stats = LEEKRL_STAT_COUNT;
    layout->options_size = (int)sizeof(leekrl_vec_options);
    layout->buffers_size = (int)sizeof(leekrl_vec_buffers);
    layout->build_size = (int)sizeof(leekrl_build);
}

int leekrl_vec_add_build(leekrl_vec *vec, const leekrl_build *build, double weight,
                         char *error, size_t error_size)
{
    leekrl_build *known;
    double *weights;
    if (error && error_size) error[0] = 0;
    if (!vec || !build || !isfinite(weight) || weight <= 0) {
        if (error && error_size) snprintf(error, error_size, "invalid argument");
        return 0;
    }
    if (!leekrl_build_validate(vec->pool, build, error, error_size)) return 0;
    known = realloc(vec->known, (size_t)(vec->n_known + 1) * sizeof(*known));
    if (known) vec->known = known;
    weights = realloc(vec->weights, (size_t)(vec->n_known + 1) * sizeof(*weights));
    if (weights) vec->weights = weights;
    if (!known || !weights) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }
    vec->known[vec->n_known] = *build;
    vec->weights[vec->n_known++] = weight;
    vec->total_weight += weight;
    return 1;
}

static int creation_error(leekrl_vec *vec, char *error, size_t size, const char *message)
{
    if (error && size) snprintf(error, size, "%s", message);
    leekrl_vec_destroy(vec);
    return 0;
}

leekrl_vec *leekrl_vec_create(const leekrl_vec_options *options,
                              const leekrl_vec_buffers *buffers, char *error,
                              size_t error_size)
{
    leekrl_vec *vec;
    int i;
    if (error && error_size) error[0] = 0;
    if (!options || !buffers || options->envs < 1 || options->envs > 1 << 20 ||
        options->threads < 1 || !(options->known_probability >= 0) ||
        options->known_probability > 1 || (unsigned)options->trim > LEEKRL_TRIM_HARMFUL ||
        !leekrl_reward_options_valid(&options->rewards) ||
        !buffers->floats || !buffers->map || !buffers->mask || !buffers->rewards ||
        !buffers->team || !buffers->first || !buffers->done || !buffers->end_rewards ||
        !buffers->episode) {
        creation_error(NULL, error, error_size, "invalid argument");
        return NULL;
    }
    vec = calloc(1, sizeof(*vec));
    if (!vec) {
        creation_error(NULL, error, error_size, "out of memory");
        return NULL;
    }
    vec->options = *options;
    vec->buffers = *buffers;
    /* Not safe to build on first use by several threads. */
    lw_area_build_masks();
    if (leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
                                             leekrl_solo_excluded_count, &vec->catalog) !=
            LEEKRL_ACTION_OK ||
        !(vec->observer = leekrl_observer_create(vec->catalog)) ||
        !(vec->pool = leekrl_build_pool_create(vec->catalog))) {
        creation_error(vec, error, error_size, "item snapshot not registered");
        return NULL;
    }
    vec->actions = leekrl_action_catalog_count(vec->catalog);
    vec->floats = leekrl_observer_float_count(vec->observer);
    vec->mask_bytes = (vec->actions + 7) / 8;
    vec->envs = calloc((size_t)options->envs, sizeof(*vec->envs));
    vec->threads = leekrl_threads_create(options->threads);
    if (!vec->envs || !vec->threads) {
        creation_error(vec, error, error_size, "out of memory");
        return NULL;
    }
    for (i = 0; i < options->envs; ++i) {
        environment *env = vec->envs[i] = calloc(1, sizeof(**vec->envs));
        if (env) env->legal = malloc(vec->actions);
        if (!env || !env->legal) {
            creation_error(vec, error, error_size, "out of memory");
            return NULL;
        }
        env->vec = vec;
        env->index = i;
        lw_generator_init(&env->generator);
        lw_generator_set_plant_ai_dispatch(&env->generator, leekrl_plant_ai, NULL);
    }
    return vec;
}

void leekrl_vec_destroy(leekrl_vec *vec)
{
    int i;
    if (!vec) return;
    leekrl_threads_destroy(vec->threads);
    for (i = 0; vec->envs && i < vec->options.envs; ++i) {
        if (!vec->envs[i]) continue;
        stop_fight(vec->envs[i]);
        free(vec->envs[i]->legal);
        free(vec->envs[i]);
    }
    free(vec->envs);
    free(vec->known);
    free(vec->weights);
    leekrl_build_pool_destroy(vec->pool);
    leekrl_observer_destroy(vec->observer);
    leekrl_action_catalog_destroy(vec->catalog);
    free(vec);
}
