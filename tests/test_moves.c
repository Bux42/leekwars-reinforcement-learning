/*
 * Moving right after a move is never needed: whatever two moves reach, one reaches it
 * too, for as few MP, and nothing else changes. Checked on clones of random fights, when
 * the first move starts with at most as many MP as the longest move, and cannot enter
 * the zone of a plant the entity has not awakened. Optional argument: number of fights.
 */
#include "leekrl/actions.h"
#include "leekrl/build.h"
#include "leekrl/item_snapshot.h"
#include "leekrl/plant_ai.h"
#include "leekrl/rules.h"
#include "lw_effect.h"
#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_generator.h"
#include "lw_outcome.h"
#include "lw_scenario.h"
#include "lw_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int current_seed;

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: seed %d: %s\n", __FILE__, __LINE__, current_seed, #x); exit(1); \
} } while (0)

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

static int distance(const LwEntity *a, const LwEntity *b)
{
    return abs(a->cell->x - b->cell->x) + abs(a->cell->y - b->cell->y);
}

static int plant_in_reach(const LwState *state, const LwEntity *actor, int mp, int *plants)
{
    int i, k, found = 0;
    for (i = 0; i < state->order.n_leeks; ++i) {
        const LwEntity *plant = state->order.leeks[i];
        int known = 0;
        if (plant->awakening_zone <= 0) continue;
        ++*plants;
        for (k = 0; k < plant->n_awakened_by; ++k) known |= plant->awakened_by[k] == actor->fight_id;
        found |= !known && distance(actor, plant) <= plant->awakening_zone + mp;
    }
    return found;
}

/* Everybody else is the same in both fights. */
static void check_others(const LwState *a, const LwState *b)
{
    int i, k;
    CHECK(a->n_entities == b->n_entities);
    for (i = 0; i < a->n_entities; ++i) {
        const LwEntity *x = a->m_entities[i], *y = b->m_entities[i];
        CHECK(x->life == y->life && x->m_total_life == y->m_total_life &&
              x->states == y->states && x->n_effects == y->n_effects &&
              x->n_awakened_by == y->n_awakened_by);
        CHECK((x->cell ? x->cell->id : -1) == (y->cell ? y->cell->id : -1));
        for (k = 0; k < x->n_effects; ++k)
            CHECK(x->effects[k]->id == y->effects[k]->id &&
                  x->effects[k]->value == y->effects[k]->value);
    }
}

/* Executes a move on a clone of a fight, which the caller releases. */
static LwEntity *move(LwFight *fight, LwState *state, LwFight *source, const leekrl_action *action)
{
    LwEntity *actor;
    int result;
    CHECK(lw_fight_step_clone(fight, state, source));
    actor = lw_fight_step_current(fight);
    CHECK(leekrl_action_execute(fight, actor, action, &result) == LEEKRL_ACTION_OK);
    return actor;
}

int main(int argc, char **argv)
{
    static LwState one, two, direct;
    static LwFight first, second, straight;
    char error[512];
    int fights = argc > 1 ? atoi(argv[1]) : 40, i;
    session *s = malloc(sizeof(*s));
    leekrl_action_catalog *catalog;
    leekrl_build_pool *pool;
    leekrl_build_options options;
    const leekrl_action *actions;
    uint8_t *legal, *after;
    uint64_t rng = 3;
    size_t n, k, m;
    long pairs = 0, shorter = 0, back = 0, plants = 0;
    if (!leekrl_item_snapshot_register(error, sizeof(error))) {
        fprintf(stderr, "item registration: %s\n", error);
        return 1;
    }
    CHECK(leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
              leekrl_solo_excluded_count, &catalog) == LEEKRL_ACTION_OK);
    actions = leekrl_action_catalog_actions(catalog);
    n = leekrl_action_catalog_count(catalog);
    pool = leekrl_build_pool_create(catalog);
    legal = malloc(n);
    after = malloc(n);
    CHECK(s && pool && legal && after);
    /* The catalog finds its own actions, and no other. */
    for (k = 0; k < n; ++k) CHECK(leekrl_action_catalog_find(catalog, &actions[k]) == k);
    CHECK(leekrl_action_catalog_find(catalog, &(leekrl_action){LEEKRL_ACTION_MOVE, 0, 0, 0}) == n);
    CHECK(leekrl_action_catalog_find(catalog, &(leekrl_action){LEEKRL_ACTION_MOVE, 0, 99, 0}) == n);
    CHECK(leekrl_action_catalog_find(catalog, &(leekrl_action){LEEKRL_ACTION_MOVE, 0, 15, 15}) == n);
    CHECK(leekrl_action_catalog_find(catalog, &(leekrl_action){LEEKRL_ACTION_WEAPON, 37, 30, 0}) == n);
    CHECK(leekrl_action_catalog_find(catalog, &(leekrl_action){LEEKRL_ACTION_CHIP, 99999, 1, 0}) == n);
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
            CHECK(leekrl_build_random(pool, &options, &rng, &build));
            leekrl_build_to_entity(&build, team + 1, &info);
            info.mp += 4;
            CHECK(lw_scenario_add_entity(&s->scenario, team, &info));
        }
        CHECK(!lw_generator_prepare_scenario(&s->generator, &s->scenario, NULL, NULL, NULL,
                                             &s->state, &s->fight, &s->outcome));
        CHECK(!lw_fight_step_start(&s->fight));
        while (!lw_fight_step_is_over(&s->fight)) {
            int plays = 0;
            CHECK(!lw_fight_step_begin_turn(&s->fight, &actor));
            if (!actor) continue;
            while (!lw_entity_is_dead(actor) && !lw_state_is_finished(&s->state) && ++plays < 12) {
                size_t choices = 0, pick;
                int result, mp = lw_entity_get_mp(actor), seen = 0;
                int skip = plant_in_reach(&s->state, actor, mp, &seen) ||
                           mp > LEEKRL_MAX_MOVE_DISTANCE;
                CHECK(leekrl_action_catalog_mask(catalog, &s->fight, actor, legal) ==
                      LEEKRL_ACTION_OK);
                for (k = 1; k <= LEEKRL_MOVE_ACTION_COUNT && !skip; ++k) {
                    LwEntity *a, *b, *c;
                    if (!legal[k] || next_random(&rng) % 40) continue;
                    a = move(&first, &one, &s->fight, &actions[k]);
                    CHECK(leekrl_action_catalog_mask(catalog, &first, a, after) ==
                          LEEKRL_ACTION_OK);
                    for (m = 1; m <= LEEKRL_MOVE_ACTION_COUNT; ++m) {
                        leekrl_action whole = actions[m];
                        size_t index;
                        if (!after[m] || next_random(&rng) % 6) continue;
                        ++pairs;
                        whole.dx += actions[k].dx;
                        whole.dy += actions[k].dy;
                        if (!whole.dx && !whole.dy) { /* back where it was */
                            ++back;
                            continue;
                        }
                        index = leekrl_action_catalog_find(catalog, &whole);
                        CHECK(index < n && legal[index]);
                        b = move(&second, &two, &first, &actions[m]);
                        c = move(&straight, &direct, &s->fight, &actions[index]);
                        check_others(&two, &direct);
                        CHECK(b->cell->id == c->cell->id);
                        plants += seen > 0;
                        CHECK(lw_entity_get_mp(c) >= lw_entity_get_mp(b));
                        shorter += lw_entity_get_mp(c) > lw_entity_get_mp(b);
                        lw_state_dispose(&two);
                        lw_state_dispose(&direct);
                    }
                    lw_state_dispose(&one);
                }
                for (k = 0; k < n; ++k) choices += legal[k];
                pick = (size_t)(next_random(&rng) % choices);
                for (k = 0; k < n; ++k)
                    if (legal[k] && pick-- == 0) break;
                if (!k) break;
                CHECK(leekrl_action_execute(&s->fight, actor, &actions[k], &result) ==
                      LEEKRL_ACTION_OK);
            }
            CHECK(!lw_fight_step_end_turn(&s->fight));
        }
        lw_state_dispose(&s->state);
        lw_scenario_dispose(&s->scenario);
    }
    printf("%ld pairs of moves: a single one always does as well, %.0f%% of the time for fewer "
           "MP; %.1f%% come back to the start, %.0f%% with a plant out of reach\n", pairs,
           100.0 * (double)shorter / (double)(pairs - back), 100.0 * (double)back / (double)pairs,
           100.0 * (double)plants / (double)(pairs - back));
    CHECK(pairs > 1000 && shorter && back && plants);
    free(legal);
    free(after);
    free(s);
    leekrl_build_pool_destroy(pool);
    leekrl_action_catalog_destroy(catalog);
    leekrl_item_snapshot_clear();
    return 0;
}
