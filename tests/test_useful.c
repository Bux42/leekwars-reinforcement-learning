/*
 * Casts trimmed from the mask must change nothing: each is executed on a clone of
 * the fight, which must then only differ by what casting costs (TP, uses,
 * cooldown, weapon in hand, random numbers). Kept casts are also executed, to
 * measure how many useless ones remain. Optional argument: number of fights.
 */
#include "leekrl/actions.h"
#include "leekrl/build.h"
#include "leekrl/item_snapshot.h"
#include "leekrl/plant_ai.h"
#include "leekrl/rules.h"
#include "lw_chip.h"
#include "lw_effect.h"
#include "lw_fight.h"
#include "lw_generator.h"
#include "lw_map.h"
#include "lw_outcome.h"
#include "lw_scenario.h"
#include "lw_state.h"
#include "lw_weapon.h"

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

static leekrl_action_catalog *catalog;

static uint64_t next_random(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static int same_stats(const LwStats *a, const LwStats *b)
{
    int i;
    for (i = 0; i < LW_STAT_COUNT; ++i)
        if (lw_stats_get_stat(a, i) != lw_stats_get_stat(b, i)) return 0;
    return 1;
}

static int same_effects(struct LwEffect *const *a, struct LwEffect *const *b, int n)
{
    int i;
    for (i = 0; i < n; ++i)
        if (a[i]->id != b[i]->id || a[i]->value != b[i]->value || a[i]->turns != b[i]->turns ||
            a[i]->state != b[i]->state) return 0;
    return 1;
}

/* Everything but the costs of casting. */
static int same_situation(const LwState *a, const LwState *b)
{
    int i;
    if (a->n_entities != b->n_entities || a->order.n_leeks != b->order.n_leeks ||
        a->order.turn != b->order.turn || a->order.position != b->order.position) return 0;
    for (i = 0; i < a->n_entities; ++i) {
        const LwEntity *x = a->m_entities[i], *y = b->m_entities[i];
        if (x->life != y->life || x->m_total_life != y->m_total_life || x->states != y->states ||
            x->used_mp != y->used_mp || (x->cell ? x->cell->id : -1) != (y->cell ? y->cell->id : -1) ||
            x->n_effects != y->n_effects || x->n_launched_effects != y->n_launched_effects ||
            x->n_awakened_by != y->n_awakened_by || !same_stats(&x->m_buff_stats, &y->m_buff_stats) ||
            !same_effects(x->effects, y->effects, x->n_effects)) return 0;
    }
    return 1;
}

/* Executes an action on a clone. Returns whether it changed the situation. */
static int changes(session *s, LwEntity *actor, const leekrl_action *action)
{
    static LwState state;
    static LwFight fight;
    LwEntity *copy;
    int result, different;
    CHECK(lw_fight_step_clone(&fight, &state, &s->fight));
    copy = lw_fight_step_current(&fight);
    CHECK(copy && lw_entity_get_fid(copy) == lw_entity_get_fid(actor));
    CHECK(leekrl_action_execute(&fight, copy, action, &result) == LEEKRL_ACTION_OK);
    different = !same_situation(&s->state, &state);
    lw_state_dispose(&state);
    return different;
}

int main(int argc, char **argv)
{
    char error[512];
    int fights = argc > 1 ? atoi(argv[1]) : 30, i;
    session *s = malloc(sizeof(*s));
    leekrl_build_pool *pool;
    leekrl_build_options options;
    const leekrl_action *actions;
    uint8_t *legal, *useful;
    uint64_t rng = 77;
    size_t n, k;
    long decisions = 0, casts = 0, kept = 0, trimmed = 0, tried = 0, kept_tried = 0, effective = 0;
    long exempt = 0, empty_kept = 0;
    if (!leekrl_item_snapshot_register(error, sizeof(error))) {
        fprintf(stderr, "item registration: %s\n", error);
        return 1;
    }
    CHECK(leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
                                               leekrl_solo_excluded_count, &catalog) ==
          LEEKRL_ACTION_OK);
    actions = leekrl_action_catalog_actions(catalog);
    n = leekrl_action_catalog_count(catalog);
    pool = leekrl_build_pool_create(catalog);
    legal = malloc(n);
    useful = malloc(n);
    CHECK(s && pool && legal && useful);
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
            info.tp += 10;
            CHECK(lw_scenario_add_entity(&s->scenario, team, &info));
        }
        CHECK(!lw_generator_prepare_scenario(&s->generator, &s->scenario, NULL, NULL, NULL,
                                             &s->state, &s->fight, &s->outcome));
        CHECK(!lw_fight_step_start(&s->fight));
        while (!lw_fight_step_is_over(&s->fight)) {
            int plays = 0;
            CHECK(!lw_fight_step_begin_turn(&s->fight, &actor));
            if (!actor) continue;
            while (!lw_entity_is_dead(actor) && !lw_state_is_finished(&s->state) && ++plays < 15) {
                size_t choices = 0, pick;
                int result;
                CHECK(leekrl_action_catalog_mask(catalog, &s->fight, actor, legal) ==
                      LEEKRL_ACTION_OK);
                CHECK(leekrl_action_catalog_useful_mask(catalog, &s->fight, actor, useful) ==
                      LEEKRL_ACTION_OK);
                ++decisions;
                for (k = 0; k < n; ++k) {
                    int cast = actions[k].kind >= LEEKRL_ACTION_WEAPON;
                    CHECK(useful[k] <= legal[k]);
                    if (!cast) CHECK(useful[k] == legal[k]);
                    if (!legal[k] || !cast) continue;
                    ++casts;
                    kept += useful[k];
                    trimmed += !useful[k];
                    if (!useful[k] && next_random(&rng) % 60 == 0) {
                        ++tried;
                        if (changes(s, actor, &actions[k])) {
                            fprintf(stderr, "seed %d: trimmed cast of item %d at (%d, %d) is "
                                    "not useless\n", current_seed, actions[k].item_id,
                                    actions[k].dx, actions[k].dy);
                            exit(1);
                        }
                    } else if (useful[k] && next_random(&rng) % 40 == 0) {
                        LwCell *cell = lw_map_get_cell_xy(s->state.map,
                            actor->cell->x + actions[k].dx, actor->cell->y + actions[k].dy);
                        ++kept_tried;
                        effective += changes(s, actor, &actions[k]);
                        empty_kept += !lw_cell_get_player(cell, s->state.map);
                    }
                }
                exempt += lw_entity_has_weapon(actor, 226);
                for (k = 0; k < n; ++k) choices += useful[k];
                pick = (size_t)(next_random(&rng) % choices);
                for (k = 0; k < n; ++k)
                    if (useful[k] && pick-- == 0) break;
                if (!k) break;
                CHECK(leekrl_action_execute(&s->fight, actor, &actions[k], &result) ==
                      LEEKRL_ACTION_OK);
            }
            CHECK(!lw_fight_step_end_turn(&s->fight));
        }
        lw_state_dispose(&s->state);
        lw_scenario_dispose(&s->scenario);
    }
    printf("%ld decisions, %ld legal casts, %ld trimmed (%.0f%%), %ld tried without effect; "
           "%ld kept casts tried, %.0f%% had an effect, %.0f%% on an empty cell\n",
           decisions, casts, trimmed, 100.0 * (double)trimmed / (double)casts, tried, kept_tried,
           100.0 * (double)effective / (double)kept_tried,
           100.0 * (double)empty_kept / (double)kept_tried);
    CHECK(tried > 1000 && kept_tried > 1000 && trimmed > casts / 2 && exempt);
    CHECK(effective > kept_tried / 3);
    free(legal);
    free(useful);
    free(s);
    leekrl_build_pool_destroy(pool);
    leekrl_action_catalog_destroy(catalog);
    leekrl_item_snapshot_clear();
    return 0;
}
