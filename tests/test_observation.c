#include "leekrl/actions.h"
#include "leekrl/battle.h"
#include "leekrl/build.h"
#include "leekrl/item_snapshot.h"
#include "leekrl/observation.h"
#include "leekrl/plant_ai.h"
#include "leekrl/rules.h"
#include "lw_bulb_template.h"
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int current_seed;

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: seed %d: %s\n", __FILE__, __LINE__, current_seed, #x); exit(1); \
} } while (0)

#define NEAR(a, b) (fabs((double)(a) - (double)(b)) < 1e-6)

enum { FLOATS = 2705, WEAPONS = 37, CHIPS = 105, LIMITED = 46, LEEK_ITEMS = 284 };

static leekrl_action_catalog *catalog;
static leekrl_observer *observer;
static leekrl_build_pool *pool;
static char error[128];

typedef struct {
    float f[FLOATS];
    uint8_t map[LEEKRL_OBS_MAP_CELLS];
} observation;

typedef struct {
    LwGenerator generator;
    LwScenario scenario;
    LwState state;
    LwFight fight;
    LwOutcome outcome;
} session;

static uint64_t next_random(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static void observe(session *s, LwEntity *actor, observation *o)
{
    memset(o, 0xAB, sizeof(*o));
    if (!leekrl_observe(observer, &s->fight, actor, o->f, o->map, error, sizeof(error))) {
        fprintf(stderr, "seed %d: %s\n", current_seed, error);
        exit(1);
    }
}

static const float *block(const observation *o, int i)
{
    return o->f + LEEKRL_OBS_ENTITIES + i * LEEKRL_OBS_ENTITY_SIZE;
}

/* Offsets written in specs/Observation.md. */
static void test_layout(void)
{
    static const int effects[] = {2, 3, 4, 5, 6, 7, 8, 13, 17, 18, 19, 20, 21, 22, 25, 26, 27,
        29, 31, 32, 37, 38, 39, 40, 41, 42, 44, 47, 48, 52, 54};
    static const int states[] = {2, 6, 9, 12, 13};
    int dx, dy, expected = 0, i;
    CHECK(LEEKRL_OBS_TYPE == 2 && LEEKRL_OBS_LIFE == 16 && LEEKRL_OBS_TOTAL_LIFE == 17);
    CHECK(LEEKRL_OBS_DX == 18 && LEEKRL_OBS_DISTANCE == 20 && LEEKRL_OBS_TP == 21);
    CHECK(LEEKRL_OBS_TOTAL_MP == 24 && LEEKRL_OBS_STRENGTH == 25 && LEEKRL_OBS_MAGIC == 30);
    CHECK(LEEKRL_OBS_FREQUENCY == 31 && LEEKRL_OBS_POWER == 32 && LEEKRL_OBS_DAMAGE_RETURN == 33);
    CHECK(LEEKRL_OBS_ABSOLUTE_SHIELD == 34 && LEEKRL_OBS_RELATIVE_SHIELD == 35);
    CHECK(LEEKRL_OBS_STATE == 36 && LEEKRL_OBS_EFFECT == 46 && LEEKRL_OBS_ZONE == 108);
    CHECK(LEEKRL_OBS_AWAKENED == 109 && LEEKRL_OBS_SUMMON_COOLDOWN == 110);
    CHECK(LEEKRL_OBS_ENTITY_SIZE == 116 && LEEKRL_OBS_ITEMS == 3 + 18 * 116);
    CHECK(leekrl_observer_weapons(observer) == WEAPONS);
    CHECK(leekrl_observer_chips(observer) == CHIPS);
    CHECK(leekrl_observer_limited(observer) == LIMITED);
    CHECK(leekrl_observer_float_count(observer) == FLOATS);
    for (i = 0; i < LEEKRL_OBS_EFFECTS; ++i) CHECK(leekrl_obs_effect_index(effects[i]) == i);
    for (i = 0; i < LEEKRL_OBS_STATES; ++i) CHECK(leekrl_obs_state_index(states[i]) == i);
    CHECK(leekrl_obs_effect_index(1) == -1 && leekrl_obs_effect_index(59) == -1);
    CHECK(leekrl_obs_state_index(3) == -1 && leekrl_obs_state_index(0) == -1);
    for (dy = -36; dy <= 36; ++dy)
        for (dx = -36; dx <= 36; ++dx) {
            if (abs(dx) + abs(dy) > 34) CHECK(leekrl_obs_map_index(dx, dy) == -1);
            else CHECK(leekrl_obs_map_index(dx, dy) == expected++);
        }
    CHECK(expected == LEEKRL_OBS_MAP_CELLS);
    CHECK(leekrl_obs_map_index(0, 0) == LEEKRL_OBS_MAP_CENTER);
    CHECK(leekrl_observer_weapon_index(observer, 37) == 0); /* pistol, lowest id */
    CHECK(leekrl_observer_weapon_index(observer, 410) == -1);
    CHECK(leekrl_observer_chip_index(observer, 444) == -1);
    CHECK(leekrl_observer_chip_index(observer, 1 << 30) == -1);
}

/* ---- Independent reading of random fights ---------------------------- */

static double amount_scale(int id)
{
    switch (id) {
    case 5: case 20: case 26: case 52: case 54: return 100;
    case 7: case 8: case 17: case 18: case 31: case 32: return 10;
    default: return 500;
    }
}

static double expected_duration(int longest, int permanent)
{
    if (permanent) return 1;
    return longest / 10.0 < 0.9 ? longest / 10.0 : 0.9;
}

static void check_effects(const float *b, const LwEntity *e)
{
    double amount[128] = {0};
    int longest[128] = {0}, permanent[128] = {0}, state_turns[32] = {0}, state_permanent[32] = {0};
    int i, id, used = 0;
    for (i = 0; i < e->n_effects; ++i) {
        const LwEffect *x = e->effects[i];
        if (x->id == 59) {
            if (x->turns == -1) state_permanent[x->state] = 1;
            else if (x->turns > state_turns[x->state]) state_turns[x->state] = x->turns;
            continue;
        }
        amount[x->id] += x->value;
        if (x->turns == -1) permanent[x->id] = 1;
        else if (x->turns > longest[x->id]) longest[x->id] = x->turns;
    }
    for (id = 0; id < 128; ++id) {
        double value = amount[id] / amount_scale(id);
        int k = leekrl_obs_effect_index(id);
        if (k < 0) {
            CHECK(amount[id] == 0);
            continue;
        }
        CHECK(NEAR(b[46 + 2 * k], value < 5 ? value : 5));
        CHECK(NEAR(b[47 + 2 * k], expected_duration(longest[id], permanent[id])));
        used += amount[id] != 0;
    }
    for (id = 0; id < 32; ++id) {
        int k = leekrl_obs_state_index(id), active = (e->states >> id) & 1;
        if (k < 0) {
            CHECK(!active);
            continue;
        }
        CHECK(b[36 + 2 * k] == active);
        CHECK(NEAR(b[37 + 2 * k], active ? expected_duration(state_turns[id],
                                                            state_permanent[id]) : 0));
    }
    (void)used;
}

static void check_entity(session *s, const float *b, LwEntity *actor, LwEntity *e)
{
    int i, type = lw_entity_is_summon(e) ? e->m_skin : 0, awakened = 0;
    int dx = e->cell->x - actor->cell->x, dy = e->cell->y - actor->cell->y;
    CHECK(b[0] == 1 && b[1] == (e->team == actor->team));
    for (i = 0; i < 14; ++i) CHECK(b[2 + i] == (i == type));
    CHECK(NEAR(b[16], (float)e->life / (float)e->m_total_life));
    CHECK(NEAR(b[17], log(1.0 + e->m_total_life) / 10));
    CHECK(NEAR(b[18], dx / 34.0) && NEAR(b[19], dy / 34.0));
    CHECK(NEAR(b[20], (abs(dx) + abs(dy)) / 34.0));
    CHECK(NEAR(b[22], lw_entity_get_stat(e, LW_STAT_TP) / 30.0));
    CHECK(NEAR(b[21], (lw_entity_get_stat(e, LW_STAT_TP) -
                       (e->awakening_zone > 0 ? 0 : e->used_tp)) / 30.0));
    CHECK(NEAR(b[24], lw_entity_get_stat(e, LW_STAT_MP) / 15.0));
    CHECK(NEAR(b[23], (lw_entity_get_stat(e, LW_STAT_MP) - e->used_mp) / 15.0));
    CHECK(NEAR(b[25], lw_entity_get_stat(e, LW_STAT_STRENGTH) / 1000.0));
    CHECK(NEAR(b[26], lw_entity_get_stat(e, LW_STAT_WISDOM) / 1000.0));
    CHECK(NEAR(b[27], lw_entity_get_stat(e, LW_STAT_AGILITY) / 1000.0));
    CHECK(NEAR(b[28], lw_entity_get_stat(e, LW_STAT_RESISTANCE) / 1000.0));
    CHECK(NEAR(b[29], lw_entity_get_stat(e, LW_STAT_SCIENCE) / 1000.0));
    CHECK(NEAR(b[30], lw_entity_get_stat(e, LW_STAT_MAGIC) / 1000.0));
    CHECK(NEAR(b[31], lw_entity_get_stat(e, LW_STAT_FREQUENCY) / 1000.0));
    CHECK(NEAR(b[32], lw_entity_get_stat(e, LW_STAT_POWER) / 100.0));
    CHECK(NEAR(b[33], lw_entity_get_stat(e, LW_STAT_DAMAGE_RETURN) / 100.0));
    CHECK(NEAR(b[34], lw_entity_get_stat(e, LW_STAT_ABSOLUTE_SHIELD) / 500.0));
    CHECK(NEAR(b[35], lw_entity_get_stat(e, LW_STAT_RELATIVE_SHIELD) / 100.0));
    check_effects(b, e);
    CHECK(NEAR(b[108], e->awakening_zone / 10.0));
    for (i = 0; i < e->n_awakened_by; ++i) awakened |= e->awakened_by[i] == actor->fight_id;
    CHECK(b[109] == (e->awakening_zone > 0 && awakened));
    if (type) {
        LwBulbTemplate *bulb = lw_bulbs_get_invocation_template(type);
        for (i = 0; i < 6; ++i) {
            int turns = i < lw_bulb_template_get_chips_count(bulb)
                ? lw_state_get_cooldown(&s->state, e, lw_bulb_template_get_chip_at(bulb, i)) : 0;
            CHECK(NEAR(b[110 + i], turns < 10 ? turns / 10.0 : 1));
        }
    } else {
        for (i = 0; i < 6; ++i) CHECK(b[110 + i] == 0);
    }
}

static void check_items(session *s, const float *o, LwEntity *leek)
{
    int i, id, ones = 0, held = 0;
    for (i = 0; i < WEAPONS + CHIPS; ++i) ones += o[i] == 1;
    CHECK(ones == leek->n_weapons + leek->n_chips);
    for (id = 0; id < 1024; ++id) {
        int k = leekrl_observer_weapon_index(observer, id), has = 0;
        if (k >= 0) {
            for (i = 0; i < leek->n_weapons; ++i) has |= lw_weapon_get_id(leek->m_weapons[i]) == id;
            CHECK(o[k] == has);
            CHECK(o[WEAPONS + CHIPS + CHIPS + k] ==
                  (leek->weapon && lw_weapon_get_id(leek->weapon) == id));
            held += o[WEAPONS + CHIPS + CHIPS + k] == 1;
        }
        k = leekrl_observer_chip_index(observer, id);
        if (k >= 0) {
            int turns = 0;
            has = 0;
            for (i = 0; i < leek->n_chips; ++i) has |= leek->m_chip_ids[i] == id;
            CHECK(o[WEAPONS + k] == has);
            if (has) turns = lw_state_get_cooldown(&s->state, leek, lw_chips_get_chip(id));
            CHECK(NEAR(o[WEAPONS + CHIPS + k], turns < 10 ? turns / 10.0 : 1));
        }
    }
    CHECK(held == (leek->weapon != NULL));
}

static struct {
    long decisions, summons, effects, states, plants, in_hand, full;
    int most_entities;
} seen;

static void check_observation(session *s, LwEntity *actor, const observation *o)
{
    LwState *state = &s->state;
    LwEntity *leeks[2] = {NULL, NULL};
    int i, j, n = state->order.n_leeks, position = 0, summons[2] = {0, 0}, dx, dy, id;
    const float *uses = o->f + LEEKRL_OBS_ITEMS + 2 * LEEK_ITEMS;
    while (state->order.leeks[position] != actor) ++position;
    CHECK(position == state->order.position);
    for (i = 0; i < 18; ++i) {
        const float *b = block(o, i);
        if (i >= n) {
            for (j = 0; j < LEEKRL_OBS_ENTITY_SIZE; ++j) CHECK(b[j] == 0);
            continue;
        }
        check_entity(s, b, actor, state->order.leeks[(position + i) % n]);
    }
    for (i = 0; i < state->n_entities; ++i) {
        LwEntity *e = state->m_entities[i];
        if (e->life <= 0) continue;
        if (lw_entity_is_summon(e)) ++summons[e->team != actor->team];
        else leeks[e->team != actor->team] = e;
        seen.effects += e->n_effects;
        seen.states += e->states != 0;
        seen.plants += e->awakening_zone > 0;
    }
    CHECK(NEAR(o->f[0], state->order.turn / 64.0));
    CHECK(NEAR(o->f[1], summons[0] / 8.0) && NEAR(o->f[2], summons[1] / 8.0));
    check_items(s, o->f + LEEKRL_OBS_ITEMS, leeks[0]);
    check_items(s, o->f + LEEKRL_OBS_ITEMS + LEEK_ITEMS, leeks[1]);
    for (id = 0; id < 1024; ++id) {
        int k = leekrl_observer_limited_index(observer, id), max = 0, has = 0;
        LwWeapon *w = lw_weapons_get_weapon(id);
        LwChip *c = lw_chips_get_chip(id);
        if (k < 0) continue;
        max = lw_attack_get_max_uses(w ? lw_weapon_get_attack(w) : lw_chip_get_attack(c));
        has = w ? lw_entity_has_weapon(actor, id) : lw_entity_get_chip(actor, id) != NULL;
        CHECK(NEAR(uses[k], has ? (max - lw_entity_get_item_uses(actor, id)) / 5.0 : 0));
        CHECK(uses[k] >= 0);
    }
    for (dy = -34; dy <= 34; ++dy) {
        for (dx = -34; dx <= 34; ++dx) {
            int index = leekrl_obs_map_index(dx, dy), expected;
            LwCell *cell;
            LwEntity *e;
            if (index < 0) {
                CHECK(!lw_map_get_cell_xy(state->map, actor->cell->x + dx, actor->cell->y + dy));
                continue;
            }
            cell = lw_map_get_cell_xy(state->map, actor->cell->x + dx, actor->cell->y + dy);
            if (!cell) expected = 0;
            else if (!cell->walkable) expected = 2;
            else if (!(e = lw_cell_get_player(cell, state->map))) expected = 1;
            else if (e == actor) expected = 3;
            else expected = (e->team == actor->team ? 4 : 6) + lw_entity_is_summon(e);
            CHECK(o->map[index] == expected);
        }
    }
    ++seen.decisions;
    seen.summons += lw_entity_is_summon(actor);
    seen.in_hand += leeks[0]->weapon != NULL;
    if (n > seen.most_entities) seen.most_entities = n;
}

static void test_random_fights(int fights)
{
    session *s = malloc(sizeof(*s));
    uint8_t *mask = malloc(leekrl_action_catalog_count(catalog));
    const leekrl_action *actions = leekrl_action_catalog_actions(catalog);
    size_t n = leekrl_action_catalog_count(catalog);
    observation *o = malloc(sizeof(*o)), *again = malloc(sizeof(*again));
    leekrl_build_options options;
    uint64_t rng = 2024;
    int i;
    CHECK(s && mask && o && again);
    leekrl_build_options_init(&options);
    for (i = 0; i < fights; ++i) {
        LwEntity *actor;
        int team;
        current_seed = i + 1;
        memset(s, 0, sizeof(*s));
        lw_generator_init(&s->generator);
        lw_generator_set_plant_ai_dispatch(&s->generator, leekrl_plant_ai, NULL);
        lw_scenario_init(&s->scenario);
        s->scenario.seed = current_seed;
        for (team = 0; team < 2; ++team) {
            LwEntityInfo info;
            leekrl_build build;
            static const int summons[] = {73, 74, 75, 76, 77, 78, 79, 80, 164, 165, 441};
            int k;
            CHECK(leekrl_build_random(pool, &options, &rng, &build));
            leekrl_build_to_entity(&build, team + 1, &info);
            /* More life, TP and summons than usual, to see more situations. */
            info.life += 3000;
            info.tp += 15;
            for (k = 0; k < (int)(sizeof(summons) / sizeof(*summons)) && k < info.n_chips; ++k) {
                int m, known = leekrl_observer_chip_index(observer, summons[k]) < 0;
                for (m = 0; m < info.n_chips; ++m) known |= info.chips[m] == summons[k];
                if (!known && next_random(&rng) % 2) info.chips[k] = summons[k];
            }
            CHECK(lw_scenario_add_entity(&s->scenario, team, &info));
        }
        CHECK(!lw_generator_prepare_scenario(&s->generator, &s->scenario, NULL, NULL, NULL,
                                             &s->state, &s->fight, &s->outcome));
        CHECK(!lw_fight_step_start(&s->fight));
        while (!lw_fight_step_is_over(&s->fight)) {
            int plays = 0;
            CHECK(!lw_fight_step_begin_turn(&s->fight, &actor));
            if (!actor) continue;
            while (!lw_entity_is_dead(actor) && !lw_state_is_finished(&s->state)) {
                uint64_t rng_before = s->state.rng_n;
                size_t legal = 0, pick, k;
                int result;
                observe(s, actor, o);
                if (next_random(&rng) % 10 == 0) {
                    check_observation(s, actor, o);
                    observe(s, actor, again);
                    CHECK(!memcmp(o, again, sizeof(*o)));
                    CHECK(s->state.rng_n == rng_before);
                }
                /* The generator keeps dead summons, and stops at 64 entities. */
                if (s->state.n_entities == LW_STATE_MAX_ENTITIES) break;
                if (++plays > 12) break;
                CHECK(leekrl_action_catalog_mask(catalog, &s->fight, actor, mask) ==
                      LEEKRL_ACTION_OK);
                /* Prefer casts that reach someone. */
                for (k = 0; k < n; ++k) {
                    LwCell *cell;
                    if (!mask[k] || actions[k].kind < LEEKRL_ACTION_WEAPON) continue;
                    cell = lw_map_get_cell_xy(s->state.map, actor->cell->x + actions[k].dx,
                                              actor->cell->y + actions[k].dy);
                    if (!lw_cell_get_player(cell, s->state.map) && next_random(&rng) % 40)
                        mask[k] = 0;
                }
                for (k = 0; k < n; ++k) legal += mask[k];
                pick = (size_t)(next_random(&rng) % legal);
                for (k = 0; k < n; ++k)
                    if (mask[k] && pick-- == 0) break;
                if (k == 0) break;
                CHECK(leekrl_action_execute(&s->fight, actor, &actions[k], &result) ==
                      LEEKRL_ACTION_OK);
            }
            if (s->state.n_entities == LW_STATE_MAX_ENTITIES) {
                ++seen.full;
                break;
            }
            CHECK(!lw_fight_step_end_turn(&s->fight));
        }
        lw_state_dispose(&s->state);
        lw_scenario_dispose(&s->scenario);
    }
    printf("%ld observations checked (%ld by summons), %ld effects, %ld entities in a state, "
           "%ld plants, %ld with a weapon in hand, %ld fights stopped at 64 entities, up to %d alive\n", seen.decisions,
           seen.summons, seen.effects, seen.states, seen.plants, seen.in_hand, seen.full,
           seen.most_entities);
    CHECK(seen.summons && seen.effects && seen.states && seen.plants && seen.in_hand);
    CHECK(seen.most_entities >= 10);
    free(mask);
    free(again);
    free(o);
    free(s);
}

/* ---- Controlled situations -------------------------------------------- */

/* Two leeks without obstacles, at the given cells. Team 0 plays first. */
static session *make_session(int cell0, int cell1)
{
    session *s = calloc(1, sizeof(*s));
    LwEntity *actor;
    int team;
    CHECK(s);
    lw_generator_init(&s->generator);
    lw_generator_set_plant_ai_dispatch(&s->generator, leekrl_plant_ai, NULL);
    lw_scenario_init(&s->scenario);
    s->scenario.seed = 3;
    s->scenario.map.present = 1;
    s->scenario.map.team1[0] = cell0;
    s->scenario.map.n_team1 = 1;
    s->scenario.map.team2[0] = cell1;
    s->scenario.map.n_team2 = 1;
    for (team = 0; team < 2; ++team) {
        LwEntityInfo info;
        lw_entity_info_init(&info);
        info.id = team + 1;
        info.level = 301;
        info.life = 1000 + 500 * team;
        info.tp = 20 + team;
        info.mp = 6 + team;
        info.ram = 10;
        info.strength = 100 - 250 * team;
        info.frequency = team ? 100 : 5000;
        info.weapons[0] = team ? 42 : 37;
        info.n_weapons = 1;
        info.chips[0] = 73; /* puny bulb */
        info.chips[1] = team ? 6 : 5;
        info.n_chips = 2;
        CHECK(lw_scenario_add_entity(&s->scenario, team, &info));
    }
    CHECK(!lw_generator_prepare_scenario(&s->generator, &s->scenario, NULL, NULL, NULL,
                                         &s->state, &s->fight, &s->outcome));
    CHECK(!lw_fight_step_start(&s->fight));
    CHECK(!lw_fight_step_begin_turn(&s->fight, &actor) && actor);
    while (actor->team != 0) {
        CHECK(!lw_fight_step_end_turn(&s->fight));
        CHECK(!lw_fight_step_begin_turn(&s->fight, &actor) && actor);
    }
    return s;
}

static void free_session(session *s)
{
    lw_state_dispose(&s->state);
    lw_scenario_dispose(&s->scenario);
    free(s);
}

static LwEntity *leek(session *s, int team)
{
    int i;
    for (i = 0; i < s->state.n_entities; ++i)
        if (s->state.m_entities[i]->team == team && !lw_entity_is_summon(s->state.m_entities[i]))
            return s->state.m_entities[i];
    CHECK(0);
    return NULL;
}

/* The same situation elsewhere changes only the cells outside the map. */
static void test_translation(void)
{
    session *a = make_session(306, 309), *b;
    LwCell *from = leek(a, 0)->cell, *to = leek(a, 1)->cell;
    LwCell *from2 = lw_map_get_cell_xy(a->state.map, from->x + 3, from->y - 2);
    LwCell *to2 = lw_map_get_cell_xy(a->state.map, to->x + 3, to->y - 2);
    observation *x = malloc(sizeof(*x)), *y = malloc(sizeof(*y));
    int i, changed = 0;
    CHECK(x && y && from2 && to2);
    b = make_session(from2->id, to2->id);
    CHECK(leek(b, 0)->cell->id == from2->id && leek(b, 1)->cell->id == to2->id);
    observe(a, leek(a, 0), x);
    observe(b, leek(b, 0), y);
    CHECK(a->state.order.turn == b->state.order.turn);
    CHECK(!memcmp(x->f, y->f, sizeof(x->f)));
    for (i = 0; i < LEEKRL_OBS_MAP_CELLS; ++i) {
        if (x->map[i] == y->map[i]) continue;
        ++changed;
        CHECK((x->map[i] == 0 && y->map[i] == 1) || (x->map[i] == 1 && y->map[i] == 0));
    }
    CHECK(changed > 0);
    free(x);
    free(y);
    free_session(a);
    free_session(b);
}

/* Each side sees the other as the enemy, at the opposite position. */
static void test_symmetry(void)
{
    session *s = make_session(306, 309);
    observation *x = malloc(sizeof(*x)), *y = malloc(sizeof(*y));
    const float *mine, *theirs;
    leekrl_action summon = {LEEKRL_ACTION_CHIP, 73, 0, -1};
    int dx, dy, result, index;
    CHECK(x && y);
    CHECK(leekrl_action_execute(&s->fight, leek(s, 0), &summon, &result) == LEEKRL_ACTION_OK);
    observe(s, leek(s, 0), x);
    observe(s, leek(s, 1), y);
    /* Order for leek 0: itself, its bulb, leek 1. For leek 1: itself, leek 0, the bulb. */
    CHECK(block(x, 0)[2] == 1 && block(x, 1)[3] == 1 && block(x, 2)[2] == 1 && block(x, 3)[0] == 0);
    CHECK(block(y, 0)[2] == 1 && block(y, 1)[2] == 1 && block(y, 2)[3] == 1);
    CHECK(block(x, 1)[1] == 1 && block(x, 2)[1] == 0 && block(y, 1)[1] == 0 && block(y, 2)[1] == 0);
    CHECK(x->f[1] == 0.125f && x->f[2] == 0 && y->f[1] == 0 && y->f[2] == 0.125f);
    mine = block(x, 2);   /* leek 1 seen by leek 0 */
    theirs = block(y, 1); /* leek 0 seen by leek 1 */
    CHECK(mine[18] == -theirs[18] && mine[19] == -theirs[19] && mine[20] == theirs[20]);
    /* Own characteristics do not depend on who looks. */
    CHECK(!memcmp(block(x, 0) + 21, block(y, 1) + 21, 87 * sizeof(float)));
    CHECK(!memcmp(block(x, 2) + 21, block(y, 0) + 21, 87 * sizeof(float)));
    CHECK(block(x, 2)[25] == -0.15f && block(x, 2)[16] == 1);
    CHECK(!memcmp(x->f + LEEKRL_OBS_ITEMS, y->f + LEEKRL_OBS_ITEMS + LEEK_ITEMS,
                  LEEK_ITEMS * sizeof(float)));
    CHECK(!memcmp(y->f + LEEKRL_OBS_ITEMS, x->f + LEEKRL_OBS_ITEMS + LEEK_ITEMS,
                  LEEK_ITEMS * sizeof(float)));
    dx = leek(s, 1)->cell->x - leek(s, 0)->cell->x;
    dy = leek(s, 1)->cell->y - leek(s, 0)->cell->y;
    CHECK(x->map[leekrl_obs_map_index(dx, dy)] == LEEKRL_CELL_ENEMY_LEEK);
    CHECK(y->map[leekrl_obs_map_index(-dx, -dy)] == LEEKRL_CELL_ENEMY_LEEK);
    CHECK(x->map[leekrl_obs_map_index(0, -1)] == LEEKRL_CELL_ALLY_SUMMON);
    CHECK(y->map[leekrl_obs_map_index(-dx, -dy - 1)] == LEEKRL_CELL_ENEMY_SUMMON);
    /* Cooldown of the bulb chip, seen on the side that used it only. */
    index = WEAPONS + CHIPS + leekrl_observer_chip_index(observer, 73);
    CHECK(x->f[LEEKRL_OBS_ITEMS + index] > 0 && y->f[LEEKRL_OBS_ITEMS + index] == 0);
    /* The bulb's point of view: the leeks keep their places in the item blocks. */
    {
        observation *z = malloc(sizeof(*z));
        LwEntity *bulb = lw_state_get_last_entity(&s->state);
        CHECK(z && lw_entity_is_summon(bulb));
        observe(s, bulb, z);
        CHECK(z->map[LEEKRL_OBS_MAP_CENTER] == LEEKRL_CELL_SELF);
        CHECK(z->map[leekrl_obs_map_index(0, 1)] == LEEKRL_CELL_ALLY_LEEK);
        CHECK(!memcmp(x->f + LEEKRL_OBS_ITEMS, z->f + LEEKRL_OBS_ITEMS,
                      2 * LEEK_ITEMS * sizeof(float)));
        free(z);
    }
    free(x);
    free(y);
    free_session(s);
}

static void add_effect(session *s, LwEntity *target, int id, int turns, double value, int item)
{
    lw_effect_create_effect(&s->state, id, turns, 1, value, 0, 0, target, target,
                            item ? lw_chip_get_attack(lw_chips_get_chip(item)) : NULL,
                            0, 1, 0, 1, 0, 0);
}

static void test_effects(void)
{
    session *s = make_session(306, 309);
    observation *o = malloc(sizeof(*o));
    LwEntity *e = leek(s, 0);
    int poison = LEEKRL_OBS_EFFECT + 2 * leekrl_obs_effect_index(13);
    int shield = LEEKRL_OBS_EFFECT + 2 * leekrl_obs_effect_index(37);
    int state = LEEKRL_OBS_STATE + 2 * leekrl_obs_state_index(LW_ENTITY_STATE_STERILE);
    CHECK(o);
    observe(s, e, o);
    CHECK(block(o, 0)[poison] == 0 && block(o, 0)[poison + 1] == 0);
    /* Total of two poisons, longest duration. */
    add_effect(s, e, 13, 4, 100, 5);
    add_effect(s, e, 13, 2, 50, 6);
    CHECK(e->n_effects == 2);
    observe(s, e, o);
    CHECK(NEAR(block(o, 0)[poison], 0.3) && NEAR(block(o, 0)[poison + 1], 0.4));
    /* Also visible on the enemy's side, on the right entity. */
    observe(s, leek(s, 1), o);
    CHECK(NEAR(block(o, 1)[poison], 0.3) && block(o, 0)[poison] == 0);
    /* Permanent beats any duration, in both orders. */
    add_effect(s, e, 37, -1, 40, 5);
    add_effect(s, e, 37, 3, 10, 6);
    observe(s, e, o);
    CHECK(NEAR(block(o, 0)[shield], 0.1) && block(o, 0)[shield + 1] == 1);
    add_effect(s, e, 13, -1, 1, 8);
    observe(s, e, o);
    CHECK(block(o, 0)[poison + 1] == 1);
    /* Limits. */
    add_effect(s, e, 13, 2, 100000, 9);
    observe(s, e, o);
    CHECK(block(o, 0)[poison] == 5);
    add_effect(s, e, 59, 25, LW_ENTITY_STATE_STERILE, 5);
    observe(s, e, o);
    CHECK(block(o, 0)[state] == 1 && NEAR(block(o, 0)[state + 1], 0.9));
    free(o);
    free_session(s);
}

static void test_uses(void)
{
    session *s = make_session(306, 309);
    observation *o = malloc(sizeof(*o));
    LwEntity *e = leek(s, 0);
    int k = leekrl_observer_limited_index(observer, 37);
    const float *uses;
    CHECK(o && k >= 0);
    uses = o->f + LEEKRL_OBS_ITEMS + 2 * LEEK_ITEMS;
    observe(s, e, o);
    CHECK(NEAR(uses[k], 4 / 5.0));
    CHECK(uses[leekrl_observer_limited_index(observer, 42)] == 0); /* the enemy's */
    lw_entity_add_item_use(e, 37);
    lw_entity_add_item_use(e, 37);
    lw_entity_add_item_use(e, 37);
    observe(s, e, o);
    CHECK(NEAR(uses[k], 1 / 5.0));
    observe(s, leek(s, 1), o);
    CHECK(uses[k] == 0 && uses[leekrl_observer_limited_index(observer, 42)] > 0);
    free(o);
    free_session(s);
}

#define REFUSED(session_, actor, text) do { \
        CHECK(!leekrl_observe(observer, &(session_)->fight, (actor), o->f, o->map, \
                              error, sizeof(error))); \
        CHECK(strstr(error, text)); \
    } while (0)

static void test_refusals(void)
{
    observation *o = malloc(sizeof(*o));
    session *s = make_session(306, 309);
    LwEntity extra[20];
    int i;
    CHECK(o);
    add_effect(s, leek(s, 1), 62, 2, 2, 0); /* multiplied stats: colossus only */
    REFUSED(s, leek(s, 0), "unlisted effect (62)");
    free_session(s);

    s = make_session(306, 309);
    add_effect(s, leek(s, 1), 59, 2, LW_ENTITY_STATE_INVINCIBLE, 0);
    REFUSED(s, leek(s, 0), "unlisted state (3)");
    free_session(s);

    s = make_session(306, 309);
    lw_entity_add_weapon(leek(s, 1), lw_weapons_get_weapon(410));
    REFUSED(s, leek(s, 0), "unlisted weapon (410)");
    free_session(s);

    s = make_session(306, 309);
    lw_entity_add_chip(leek(s, 0), lw_chips_get_chip(84));
    REFUSED(s, leek(s, 1), "unlisted chip (84)");
    free_session(s);

    s = make_session(306, 309);
    for (i = 0; i < 17; ++i) {
        extra[i] = *leek(s, 0);
        CHECK(lw_order_add_entity(&s->state.order, &extra[i]));
    }
    REFUSED(s, leek(s, 0), "too many entities (19)");
    s->state.order.n_leeks = 3;
    REFUSED(s, leek(s, 0), "living entity not in the turn order");
    s->state.order.n_leeks = 2;
    observe(s, leek(s, 0), o);
    CHECK(!leekrl_observe(observer, &s->fight, &extra[0], o->f, o->map, error, sizeof(error)));
    CHECK(!leekrl_observe(observer, &s->fight, NULL, o->f, o->map, error, sizeof(error)));
    lw_stats_set_stat(&leek(s, 0)->m_base_stats, LW_STAT_STRENGTH, 5);
    leek(s, 0)->life = 0;
    REFUSED(s, leek(s, 0), "cannot observe");
    free_session(s);
    free(o);
}

/* Through the environment: same seed and actions, same bytes. */
static void test_determinism(void)
{
    leekrl_build_options options;
    LwEntityInfo info[2];
    leekrl_battle_options battle_options;
    uint64_t rng = 5, hash[2];
    int i, run;
    leekrl_build_options_init(&options);
    for (i = 0; i < 2; ++i) {
        leekrl_build build;
        CHECK(leekrl_build_random(pool, &options, &rng, &build));
        leekrl_build_to_entity(&build, i + 1, &info[i]);
    }
    leekrl_battle_options_init(&battle_options);
    for (run = 0; run < 2; ++run) {
        leekrl_battle *battle = NULL;
        leekrl_env *env = NULL;
        size_t n = leekrl_action_catalog_count(catalog), size, k;
        rng = 77;
        hash[run] = UINT64_C(0xCBF29CE484222325);
        CHECK(leekrl_battle_create(&info[0], &info[1], &battle_options, &battle) ==
              LEEKRL_ENV_OK);
        size = leekrl_battle_observation_size(battle);
        CHECK(size == FLOATS + LEEKRL_OBS_MAP_CELLS);
        CHECK(leekrl_env_create(size, n, leekrl_battle_callbacks(), battle, &env) ==
              LEEKRL_ENV_OK);
        CHECK(leekrl_env_reset(env, 11) == LEEKRL_ENV_OK);
        for (i = 0; i < 300; ++i) {
            const unsigned char *bytes = (const unsigned char *)leekrl_env_observation(env);
            const uint8_t *mask = leekrl_env_action_mask(env);
            leekrl_step_result result;
            size_t legal = 0, pick;
            for (k = 0; k < size * sizeof(float); ++k)
                hash[run] = (hash[run] ^ bytes[k]) * UINT64_C(0x100000001B3);
            for (k = 0; k < n; ++k) legal += mask[k];
            pick = (size_t)(next_random(&rng) % legal);
            for (k = 0; k < n; ++k)
                if (mask[k] && pick-- == 0) break;
            CHECK(leekrl_env_step(env, k, &result) == LEEKRL_ENV_OK);
            if (result.terminated) break;
        }
        CHECK(!leekrl_battle_error(battle)[0]);
        leekrl_env_destroy(env);
        leekrl_battle_destroy(battle);
    }
    CHECK(hash[0] == hash[1]);
}

static void speed(void)
{
    session *s = make_session(306, 309);
    observation *o = malloc(sizeof(*o));
    clock_t start;
    int i, n = 50000;
    CHECK(o);
    start = clock();
    for (i = 0; i < n; ++i)
        CHECK(leekrl_observe(observer, &s->fight, leek(s, 0), o->f, o->map, error, sizeof(error)));
    printf("%.2f us per observation (2 entities)\n",
           1e6 * (double)(clock() - start) / CLOCKS_PER_SEC / n);
    free(o);
    free_session(s);
}

int main(int argc, char **argv)
{
    char message[512];
    if (!leekrl_item_snapshot_register(message, sizeof(message))) {
        fprintf(stderr, "item registration: %s\n", message);
        return 1;
    }
    CHECK(leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
                                               leekrl_solo_excluded_count, &catalog) ==
          LEEKRL_ACTION_OK);
    observer = leekrl_observer_create(catalog);
    pool = leekrl_build_pool_create(catalog);
    CHECK(observer && pool);
    CHECK(!leekrl_observer_create(NULL));
    test_layout();
    test_translation();
    test_symmetry();
    test_effects();
    test_uses();
    test_refusals();
    test_determinism();
    test_random_fights(argc > 1 ? atoi(argv[1]) : 40);
    speed();
    leekrl_build_pool_destroy(pool);
    leekrl_observer_destroy(observer);
    leekrl_action_catalog_destroy(catalog);
    leekrl_item_snapshot_clear();
    return 0;
}
