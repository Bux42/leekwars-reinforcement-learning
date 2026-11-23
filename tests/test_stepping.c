/*
 * Random fights, played three ways from the same seed and the same policy:
 * through the engine's AI callback, step by step, and step by step with the
 * fight replaced by a clone at random decisions. All must end in the same
 * state. Optional arguments: number of fights, first seed.
 */
#include "leekrl/actions.h"
#include "leekrl/item_snapshot.h"
#include "leekrl/plant_ai.h"
#include "lw_chip.h"
#include "lw_effect.h"
#include "lw_fight.h"
#include "lw_generator.h"
#include "lw_map.h"
#include "lw_outcome.h"
#include "lw_scenario.h"
#include "lw_state.h"
#include "lw_team.h"
#include "lw_weapon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: seed %d: %s\n", __FILE__, __LINE__, current_seed, #x); exit(1); \
} } while (0)

static int current_seed;

typedef struct {
    uint64_t rng;
    const leekrl_action *actions;
    size_t count;
} policy;

typedef struct {
    LwState state;
    LwFight fight;
} session;

static struct {
    long fights, decisions, summon_decisions, awakenings, plant_casts, clones, draws;
    long full_logs, dead_on_own_turn;
} stats;

static uint64_t next_random(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/* Play one random legal action. Returns 0 to end the turn. */
static int act(policy *p, LwFight *fight, LwEntity *actor)
{
    LwState *state = lw_fight_get_state(fight);
    int tries, result;
    uint8_t legal;
    if (lw_entity_is_dead(actor) || lw_state_is_finished(state) ||
        next_random(&p->rng) % 25 == 0) return 0;
    for (tries = 0; tries < 80; ++tries) {
        const leekrl_action *action = &p->actions[next_random(&p->rng) % p->count];
        CHECK(leekrl_action_mask(fight, actor, action, 1, &legal) == LEEKRL_ACTION_OK);
        if (!legal) continue;
        CHECK(leekrl_action_execute(fight, actor, action, &result) == LEEKRL_ACTION_OK);
        return 1;
    }
    return 0;
}

static int dispatch(LwFight *fight, LwEntity *actor, void *ai_file, int turn, void *userdata)
{
    (void)ai_file;
    (void)turn;
    while (act(userdata, fight, actor)) {}
    return 0;
}

static int plant_dispatch(LwFight *fight, LwEntity *plant, LwEntity *trigger,
                          int64_t ai_id, int turn, void *userdata)
{
    LwState *state = lw_fight_get_state(fight);
    int tp = lw_entity_get_tp(plant);
    int result = leekrl_plant_ai(fight, plant, trigger, ai_id, turn, userdata);
    CHECK(ai_id == LEEKRL_SUMMON_AI_ID);
    CHECK(!lw_state_get_fatal_error(state));
    ++stats.awakenings;
    stats.plant_casts += lw_entity_get_tp(plant) < tp;
    return result;
}

/* ---- State comparison ---------------------------------------------- */

static void mix(uint64_t *hash, int64_t value)
{
    *hash = (*hash ^ (uint64_t)value) * UINT64_C(0x100000001B3);
    *hash ^= *hash >> 29;
}

static void mix_double(uint64_t *hash, double value)
{
    int64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    mix(hash, bits);
}

static int fid_of(const LwEntity *e) { return e ? lw_entity_get_fid(e) : -1; }

static void mix_effect(uint64_t *hash, const LwEffect *e)
{
    int i;
    mix(hash, e->id);
    mix(hash, e->turns);
    mix_double(hash, e->aoe);
    mix_double(hash, e->value1);
    mix_double(hash, e->value2);
    mix(hash, e->critical);
    mix_double(hash, e->critical_power);
    mix(hash, fid_of(e->caster));
    mix(hash, fid_of(e->target));
    mix(hash, e->attack ? e->attack->item_id : -1);
    mix_double(hash, e->jet);
    for (i = 0; i < LW_STAT_COUNT; ++i) mix(hash, lw_stats_get_stat(&e->stats, i));
    mix(hash, e->log_id);
    mix_double(hash, e->erosion_rate);
    mix(hash, e->value);
    mix(hash, e->previous_effect_total_value);
    mix(hash, e->target_count);
    mix(hash, e->propagate);
    mix(hash, e->modifiers);
    mix(hash, e->state);
}

static void mix_entity(uint64_t *hash, const LwEntity *e)
{
    int i;
    mix(hash, e->type);
    mix(hash, e->m_id);
    mix(hash, e->fight_id);
    mix(hash, e->m_level);
    mix(hash, e->team);
    mix(hash, e->life);
    mix(hash, e->m_total_life);
    mix(hash, e->m_initial_life);
    mix(hash, e->used_tp);
    mix(hash, e->used_mp);
    mix(hash, e->cell ? e->cell->id : -1);
    for (i = 0; i < LW_STAT_COUNT; ++i) {
        mix(hash, lw_stats_get_stat(&e->m_base_stats, i));
        mix(hash, lw_stats_get_stat(&e->m_buff_stats, i));
    }
    mix(hash, e->n_effects);
    for (i = 0; i < e->n_effects; ++i) mix_effect(hash, e->effects[i]);
    mix(hash, e->n_launched_effects);
    for (i = 0; i < e->n_launched_effects; ++i) mix_effect(hash, e->launched_effects[i]);
    mix(hash, e->n_passive_effects);
    mix(hash, e->n_cooldowns);
    for (i = 0; i < e->n_cooldowns; ++i) {
        mix(hash, e->m_cooldown[i].key);
        mix(hash, e->m_cooldown[i].value);
    }
    mix(hash, e->n_item_uses);
    for (i = 0; i < e->n_item_uses; ++i) {
        mix(hash, e->item_uses[i].key);
        mix(hash, e->item_uses[i].value);
    }
    mix(hash, e->states);
    mix(hash, e->n_chips);
    for (i = 0; i < e->n_chips; ++i) mix(hash, e->m_chip_ids[i]);
    mix(hash, e->n_weapons);
    mix(hash, e->weapon ? lw_weapon_get_id(e->weapon) : -1);
    mix(hash, e->m_static);
    mix(hash, e->resurrected);
    mix(hash, e->m_birth_turn);
    mix(hash, e->m_has_moved);
    mix(hash, e->orientation);
    mix(hash, fid_of(e->m_owner));
    mix(hash, e->awakening_zone);
    mix(hash, e->n_awakened_by);
    for (i = 0; i < e->n_awakened_by; ++i) mix(hash, e->awakened_by[i]);
    mix(hash, e->has_summon_ai);
    mix(hash, e->summon_ai_id);
}

static uint64_t digest(session *s)
{
    uint64_t hash = UINT64_C(0xCBF29CE484222325);
    LwState *state = &s->state;
    LwActionStream *log = &state->actions.stream;
    int i, j;
    mix(&hash, (int64_t)state->rng_n);
    mix(&hash, state->m_state);
    mix(&hash, state->m_next_entity_id);
    mix(&hash, state->last_turn);
    mix(&hash, state->fatal_error);
    mix(&hash, s->fight.m_winteam);
    mix(&hash, state->order.turn);
    mix(&hash, state->order.position);
    mix(&hash, state->order.n_leeks);
    for (i = 0; i < state->order.n_leeks; ++i) mix(&hash, fid_of(state->order.leeks[i]));
    mix(&hash, state->n_entities);
    for (i = 0; i < state->n_entities; ++i) mix_entity(&hash, state->m_entities[i]);
    for (i = 0; i < state->n_teams; ++i) {
        LwTeam *team = state->teams[i];
        mix(&hash, team->n_entities);
        for (j = 0; j < team->n_entities; ++j) mix(&hash, fid_of(team->entities[j]));
        mix(&hash, team->n_cooldowns);
        for (j = 0; j < team->n_cooldowns; ++j) {
            mix(&hash, team->cooldowns[j].key);
            mix(&hash, team->cooldowns[j].value);
        }
    }
    for (i = 0; i < state->map->nb_cells; ++i) {
        mix(&hash, state->map->cells[i].walkable);
        mix(&hash, state->map->entity_at_cell[i]);
    }
    mix(&hash, log->n_entries);
    mix(&hash, log->next_effect_log_id);
    for (i = 0; i < log->n_entries; ++i) {
        const LwActionLog *entry = &log->entries[i];
        mix(&hash, entry->type);
        mix(&hash, entry->n_args);
        for (j = 0; j < entry->n_args; ++j) mix(&hash, entry->v[j]);
        mix(&hash, entry->extra_len);
        for (j = 0; j < entry->extra_len; ++j) mix(&hash, log->extra[entry->extra_offset + j]);
    }
    return hash;
}

/* ---- The three ways to play a fight -------------------------------- */

static void prepare(LwGenerator *generator, LwScenario *scenario, session *s,
                    LwOutcome *outcome)
{
    CHECK(!lw_generator_prepare_scenario(generator, scenario, NULL, NULL, NULL,
                                         &s->state, &s->fight, outcome));
}

static uint64_t play_callback(LwScenario *scenario, policy p, LwOutcome *outcome)
{
    session *s = calloc(1, sizeof(*s));
    LwGenerator generator;
    uint64_t result;
    CHECK(s);
    lw_generator_init(&generator);
    lw_generator_set_ai_dispatch(&generator, dispatch, &p);
    lw_generator_set_plant_ai_dispatch(&generator, plant_dispatch, NULL);
    CHECK(!lw_generator_run_scenario(&generator, scenario, NULL, NULL, NULL, &s->state,
                                     &s->fight, outcome));
    /* run_scenario also marks the fight as finished; the step API leaves that
     * to its caller. */
    CHECK(s->state.m_state == LW_FIGHT_STATE_FINISHED);
    result = digest(s);
    stats.draws += s->fight.m_winteam == -1;
    stats.full_logs += s->state.actions.stream.n_entries == LW_LOG_MAX_ENTRIES;
    lw_state_dispose(&s->state);
    free(s);
    return result;
}

static uint64_t finish(session *s)
{
    CHECK(lw_fight_step_end_turn(&s->fight) != 0);
    CHECK(!lw_fight_step_finish(&s->fight, 0));
    lw_fight_finish_fight(&s->fight);
    return digest(s);
}

/* Play until the end of the fight, from any decision. */
static void play_rest(session *s, policy *p)
{
    LwEntity *actor = lw_fight_step_current(&s->fight);
    while (!lw_fight_step_is_over(&s->fight)) {
        if (!actor) CHECK(!lw_fight_step_begin_turn(&s->fight, &actor));
        if (!actor) continue;
        while (act(p, &s->fight, actor)) {}
        CHECK(!lw_fight_step_end_turn(&s->fight));
        actor = NULL;
    }
}

static uint64_t play_steps(LwScenario *scenario, policy p, LwOutcome *outcome,
                           int clone_rate)
{
    session *s = calloc(1, sizeof(*s));
    LwGenerator generator;
    LwEntity *actor, *unused;
    uint64_t result, choice = p.rng ^ 77;
    CHECK(s);
    lw_generator_init(&generator);
    lw_generator_set_plant_ai_dispatch(&generator, plant_dispatch, NULL);
    prepare(&generator, scenario, s, outcome);
    CHECK(!lw_fight_step_start(&s->fight));
    CHECK(lw_fight_step_end_turn(&s->fight) != 0);
    while (!lw_fight_step_is_over(&s->fight)) {
        CHECK(!lw_fight_step_begin_turn(&s->fight, &actor));
        CHECK(lw_fight_step_current(&s->fight) == actor);
        if (!actor) continue;
        CHECK(lw_fight_step_begin_turn(&s->fight, &unused) != 0);
        CHECK(lw_fight_step_finish(&s->fight, 0) != 0);
        for (;;) {
            if (clone_rate && next_random(&choice) % (uint64_t)clone_rate == 0) {
                session *copy = calloc(1, sizeof(*copy));
                uint64_t before = digest(s);
                CHECK(copy && lw_fight_step_clone(&copy->fight, &copy->state, &s->fight));
                CHECK(digest(copy) == before);
                ++stats.clones;
                if (next_random(&choice) % 2) {
                    /* Play the clone to its end: the original must not move. */
                    policy q = p;
                    uint64_t end;
                    play_rest(copy, &q);
                    end = finish(copy);
                    CHECK(digest(s) == before);
                    lw_state_dispose(&copy->state);
                    free(copy);
                    play_rest(s, &p);
                    result = finish(s);
                    CHECK(result == end);
                    lw_state_dispose(&s->state);
                    free(s);
                    return result;
                }
                /* Drop the original and go on with the clone. */
                lw_state_dispose(&s->state);
                free(s);
                s = copy;
                actor = lw_fight_step_current(&s->fight);
                CHECK(actor);
            }
            if (!clone_rate) {
                ++stats.decisions;
                stats.summon_decisions += lw_entity_is_summon(actor);
            }
            if (!act(&p, &s->fight, actor)) break;
            stats.dead_on_own_turn += !clone_rate && lw_entity_is_dead(actor);
        }
        CHECK(!lw_fight_step_end_turn(&s->fight));
    }
    result = finish(s);
    lw_state_dispose(&s->state);
    free(s);
    return result;
}

/* ---- Random builds -------------------------------------------------- */

static int in_list(const int *list, int n, int id)
{
    while (n--) if (list[n] == id) return 1;
    return 0;
}

static void random_build(LwEntityInfo *info, int id, uint64_t *rng)
{
    static const int summons[] = {73, 74, 75, 76, 77, 78, 79, 80, 164, 165, 441};
    int n, i;
    LwWeapon **weapons = lw_weapons_get_templates(&n);
    LwChip **chips;
    lw_entity_info_init(info);
    info->id = id;
    info->level = 301;
    info->life = 1500 + (int)(next_random(rng) % 2500);
    info->tp = 12 + (int)(next_random(rng) % 15);
    info->mp = 4 + (int)(next_random(rng) % 12);
    info->strength = (int)(next_random(rng) % 500);
    info->agility = (int)(next_random(rng) % 400);
    info->wisdom = (int)(next_random(rng) % 400);
    info->resistance = (int)(next_random(rng) % 400);
    info->science = (int)(next_random(rng) % 400);
    info->magic = (int)(next_random(rng) % 400);
    info->frequency = 100 + (int)(next_random(rng) % 200);
    info->cores = 10;
    info->ram = 32;
    while (info->n_weapons < 4) {
        int item = lw_weapon_get_id(weapons[next_random(rng) % (uint64_t)n]);
        if (!in_list(info->weapons, info->n_weapons, item))
            info->weapons[info->n_weapons++] = item;
    }
    /* Summons are over-represented on purpose. */
    for (i = 0; i < (int)(sizeof(summons) / sizeof(*summons)); ++i)
        if (next_random(rng) % 2 && lw_chips_get_chip(summons[i]))
            info->chips[info->n_chips++] = summons[i];
    chips = lw_chips_get_templates(&n);
    while (info->n_chips < 28) {
        LwChip *chip = chips[next_random(rng) % (uint64_t)n];
        int item = lw_chip_get_id(chip);
        /* Instant kills would end most fights on turn 1. */
        if (lw_attack_get_effect_parameters_by_type(lw_chip_get_attack(chip),
                                                    LW_EFFECT_TYPE_KILL)) continue;
        if (!in_list(info->chips, info->n_chips, item)) info->chips[info->n_chips++] = item;
    }
}

/* Keep movements, end turn excluded, and the items of these builds and of the
 * summons: random sampling then often finds a legal action. */
static size_t filter_actions(const leekrl_action_catalog *catalog, const LwScenario *scenario,
                             leekrl_action *out)
{
    const leekrl_action *actions = leekrl_action_catalog_actions(catalog);
    size_t i, n = 0, count = leekrl_action_catalog_count(catalog);
    for (i = 0; i < count; ++i) {
        int team, keep = actions[i].kind == LEEKRL_ACTION_MOVE;
        for (team = 0; team < 2 && !keep; ++team) {
            const LwEntityInfo *info = &scenario->entities[team][0];
            if (actions[i].kind == LEEKRL_ACTION_WEAPON)
                keep = in_list(info->weapons, info->n_weapons, actions[i].item_id);
            else if (actions[i].kind == LEEKRL_ACTION_CHIP)
                keep = in_list(info->chips, info->n_chips, actions[i].item_id) ||
                       actions[i].item_id < 40 || actions[i].item_id == 85;
        }
        /* Range 50 items would dominate the sample. */
        if (abs(actions[i].dx) + abs(actions[i].dy) > 14 && actions[i].kind != LEEKRL_ACTION_MOVE)
            keep = 0;
        if (keep) out[n++] = actions[i];
    }
    return n;
}

int main(int argc, char **argv)
{
    char error[512];
    int fights = argc > 1 ? atoi(argv[1]) : 200, first = argc > 2 ? atoi(argv[2]) : 1, i;
    leekrl_action_catalog *catalog;
    leekrl_action *filtered;
    LwScenario *scenario = malloc(sizeof(*scenario));
    LwOutcome *outcome = malloc(sizeof(*outcome));
    if (!leekrl_item_snapshot_register(error, sizeof(error))) {
        fprintf(stderr, "item registration: %s\n", error);
        return 1;
    }
    CHECK(scenario && outcome);
    CHECK(leekrl_action_catalog_create(&catalog) == LEEKRL_ACTION_OK);
    filtered = malloc(leekrl_action_catalog_count(catalog) * sizeof(*filtered));
    CHECK(filtered);
    for (i = 0; i < fights; ++i) {
        LwEntityInfo info;
        policy p;
        uint64_t rng, expected;
        int team;
        current_seed = first + i;
        rng = (uint64_t)current_seed;
        lw_scenario_init(scenario);
        scenario->seed = current_seed;
        scenario->max_turns = LW_FIGHT_MAX_TURNS;
        for (team = 0; team < 2; ++team) {
            random_build(&info, 1 + team, &rng);
            CHECK(lw_scenario_add_entity(scenario, team, &info));
        }
        p.rng = next_random(&rng);
        p.actions = filtered;
        p.count = filter_actions(catalog, scenario, filtered);
        expected = play_callback(scenario, p, outcome);
        CHECK(play_steps(scenario, p, outcome, 0) == expected);
        /* Early clones, then clones anywhere in the fight. */
        CHECK(play_steps(scenario, p, outcome, 40) == expected);
        CHECK(play_steps(scenario, p, outcome, 1000) == expected);
        lw_scenario_dispose(scenario);
        ++stats.fights;
    }
    printf("%ld fights, %ld decisions (%ld by summons), %ld awakenings (%ld with casts), "
           "%ld clones, %ld draws, %ld full logs, %ld deaths on own turn\n",
           stats.fights, stats.decisions, stats.summon_decisions, stats.awakenings,
           stats.plant_casts, stats.clones, stats.draws, stats.full_logs,
           stats.dead_on_own_turn);
    CHECK(stats.summon_decisions && stats.awakenings && stats.plant_casts && stats.clones);
    free(filtered);
    free(outcome);
    free(scenario);
    leekrl_action_catalog_destroy(catalog);
    leekrl_item_snapshot_clear();
    return 0;
}
