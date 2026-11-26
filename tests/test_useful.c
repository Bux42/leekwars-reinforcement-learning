/*
 * Casts trimmed from the mask.
 *
 * Useless ones must change nothing: each is executed on a clone of the fight,
 * which must then only differ by what casting costs (TP, uses, cooldown, weapon
 * in hand, random numbers). Or exactly what a kept cast of the same item does: one
 * is kept of those that only matter to their caster. Kept casts are also executed,
 * to measure how many useless ones remain.
 *
 * Harmful ones must be those of the list agreed on, written a second time here,
 * no more and no less. They must never help: the fight is followed with and
 * without the cast, where everyone only ends their turns. Casts kept by an
 * exception are followed too: some must help, or the exception is pointless.
 *
 * Optional arguments: number of fights, seed of the first one.
 */
#include "leekrl/actions.h"
#include "leekrl/build.h"
#include "leekrl/item_snapshot.h"
#include "leekrl/plant_ai.h"
#include "leekrl/rules.h"
#include "lw_area.h"
#include "lw_attack.h"
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

/* ---- Items ------------------------------------------------------------- */

enum {
    /* Weapons. */
    GAZOR = 48, J_LASER = 115, ILLICIT_GRENADE_LAUNCHER = 116, MYSTERIOUS_ELECTRISOR = 117,
    UNBRIDLED_GAZOR = 118, REVOKED_M_LASER = 119, EXPLORER_RIFLE = 175, DARK_KATANA = 187,
    ENHANCED_LIGHTNINGER = 225, UNSTABLE_DESTROYER = 226, QUANTUM_RIFLE = 428,
    PLUTONIUM_BAZOOKA = 506,
    /* Chips. */
    SHOCK = 1, ICE = 2, BANDAGE = 3, CURE = 4, FLASH = 6, DRIP = 10, SPARK = 18, PEBBLE = 19,
    REGENERATION = 35, ARMORING = 67, PUNY_BULB = 73, FIRE_BULB = 74, ICED_BULB = 77,
    LIGHTNING_BULB = 78, DEVIL_STRIKE = 85, SLOW_DOWN = 92, BALL_AND_CHAIN = 93,
    TRANQUILIZER = 94, SOPORIFIC = 95, TOXIN = 98, PLAGUE = 99, FRACTURE = 106,
    PUNISHMENT = 114, WIZARD_BULB = 142, PLASMA = 143, COVID = 152, ELEVATION = 154,
    KNOWLEDGE = 155, WIZARDRY = 156, THERAPY = 158, MUTATION = 159, DESINTEGRATION = 160,
    TRANSMUTATION = 161, CRUSHING = 169, BRAINWASHING = 170, BRAMBLE = 172,
    MANUMISSION = 174, HEMORRHAGE = 273, PRISM = 276, SUPERINFECTION = 443,
    LIBERATION = 34, ANTIDOTE = 110, ALTERATION = 141
};

/* Why an item is on the list. */
typedef enum { NOT_LISTED, DAMAGES, POISONS, ERODES, COSTS, WEAKENS, HELPS } reason;

static reason listed(int item)
{
    switch (item) {
    case SHOCK: case PEBBLE: case ICE: case SPARK: case FLASH: case DEVIL_STRIKE: case PLASMA:
    case UNBRIDLED_GAZOR:
        return DAMAGES;
    case TOXIN: case PLAGUE: case GAZOR:
        return POISONS;
    case DESINTEGRATION:
        return ERODES;
    case PUNISHMENT: case DARK_KATANA:
        return COSTS;
    case TRANQUILIZER: case SOPORIFIC: case PLUTONIUM_BAZOOKA: case SLOW_DOWN:
    case BALL_AND_CHAIN: case FRACTURE: case CRUSHING: case BRAINWASHING: case HEMORRHAGE:
        return WEAKENS;
    case BANDAGE: case CURE: case DRIP: case THERAPY: case REGENERATION: case EXPLORER_RIFLE:
    case ARMORING: case ELEVATION: case MUTATION: case TRANSMUTATION: case KNOWLEDGE:
    case PRISM: case WIZARDRY: case MANUMISSION:
        return HELPS;
    default:
        return NOT_LISTED;
    }
}

/* Its effect replaces the one its target already has. */
static int replaces(int item)
{
    return item == TRANQUILIZER || item == HEMORRHAGE || item == KNOWLEDGE || item == PRISM ||
           item == WIZARDRY;
}

static int gives_life(int item)
{
    return listed(item) == HELPS && item != KNOWLEDGE && item != PRISM && item != WIZARDRY &&
           item != MANUMISSION;
}

/* Exceptions: why a cast of the list is kept. */
typedef enum {
    NO_EXCEPTION, CRITICAL_HITS, SUMMON_ITSELF, REPLACED, KILLS, INFECTED,
    FED_DAMAGE, FED_POISON, FED_EROSION, EXCEPTIONS
} exception;

static const char *const exception_names[] = {
    "none", "critical hits heal", "summon hurting itself", "replaces an effect",
    "life given to an enemy while kills are rewarded",
    "magic given to an enemy during covid", "damage to itself feeds a weapon", "poison on itself feeds a weapon",
    "maximum life lost feeds a weapon"
};

/* Casts that could look harmful and must stay. */
static int special(int item)
{
    reason why = listed(item);
    return item == COVID || item == BRAMBLE || item == SUPERINFECTION || why == DAMAGES ||
           why == POISONS || why == ERODES;
}

/* Useless on some of the entities they reach: always tried, they are rare. */
static int conditional(int item)
{
    return item == ALTERATION || item == DESINTEGRATION || item == SUPERINFECTION ||
           item == ANTIDOTE || item == LIBERATION || item == COVID;
}

static int holds(LwEntity *entity, int weapon) { return lw_entity_has_weapon(entity, weapon); }

static int legal_item(const leekrl_action *action, LwAttack **attack)
{
    if (action->kind == LEEKRL_ACTION_WEAPON)
        *attack = lw_weapon_get_attack(lw_weapons_get_weapon(action->item_id));
    else
        *attack = lw_chip_get_attack(lw_chips_get_chip(action->item_id));
    return action->item_id;
}

typedef struct { int caster, allies, enemies, n; LwEntity *entities[64]; } reached;

static void reach(session *s, LwEntity *actor, const leekrl_action *action, reached *r)
{
    LwAttack *attack;
    LwCell cells[64], *target = lw_map_get_cell_xy(s->state.map, actor->cell->x + action->dx,
                                                   actor->cell->y + action->dy);
    int i, n;
    legal_item(action, &attack);
    memset(r, 0, sizeof(*r));
    n = lw_area_get_area(&attack->area, s->state.map, actor->cell, target, actor, cells, 64);
    for (i = 0; i < n; ++i) {
        LwEntity *entity = lw_cell_get_player(&cells[i], s->state.map);
        if (!entity || lw_entity_is_dead(entity)) continue;
        r->entities[r->n++] = entity;
        if (entity == actor) r->caster = 1;
        else if (lw_entity_get_team(entity) == lw_entity_get_team(actor)) ++r->allies;
        else ++r->enemies;
    }
}

static int lacks_life(const LwEntity *entity)
{
    return lw_entity_get_life(entity) < lw_entity_get_total_life(entity) &&
           !lw_entity_has_state(entity, LW_ENTITY_STATE_UNHEALABLE);
}

static int only_heals(int item)
{
    return item == BANDAGE || item == CURE || item == DRIP || item == THERAPY ||
           item == REGENERATION || item == EXPLORER_RIFLE;
}

/* Whether a cast is on the list, and which exception keeps it. */
static int on_the_list(session *s, LwEntity *actor, const leekrl_action *action,
                       exception *kept)
{
    reason why = listed(action->item_id);
    reached r;
    int i, harmful = 0;
    *kept = NO_EXCEPTION;
    if (why == NOT_LISTED) return 0;
    reach(s, actor, action, &r);
    /* Those a heal, or a loss of maximum life, does nothing to do not count. */
    if (only_heals(action->item_id) || why == ERODES) {
        r.caster = r.allies = r.enemies = 0;
        for (i = 0; i < r.n; ++i) {
            if (why == ERODES ? lw_entity_get_life(r.entities[i]) ==
                                lw_entity_get_total_life(r.entities[i])
                              : !lacks_life(r.entities[i])) continue;
            if (r.entities[i] == actor) r.caster = 1;
            else if (lw_entity_get_team(r.entities[i]) == lw_entity_get_team(actor)) ++r.allies;
            else ++r.enemies;
        }
    }
    switch (why) {
    case DAMAGES: case POISONS: case ERODES:
        harmful = r.caster && !r.allies && !r.enemies;
        break;
    case COSTS:
        harmful = !r.n;
        break;
    case WEAKENS:
        harmful = (r.caster || r.allies) && !r.enemies;
        break;
    case HELPS:
        harmful = r.enemies && !r.caster && !r.allies;
        break;
    default:
        break;
    }
    if (!harmful) return 0;
    if (holds(actor, UNSTABLE_DESTROYER) && lw_entity_get_agility(actor) > 0 &&
        lacks_life(actor)) {
        *kept = CRITICAL_HITS;
    } else if (why == WEAKENS || why == HELPS) {
        for (i = 0; i < r.n && replaces(action->item_id); ++i)
            if (lw_entity_has_effect(r.entities[i], action->item_id)) *kept = REPLACED;
        for (i = 0; i < s->state.n_entities && gives_life(action->item_id); ++i)
            if (!lw_entity_is_dead(s->state.m_entities[i]) &&
                holds(s->state.m_entities[i], ENHANCED_LIGHTNINGER)) *kept = KILLS;
        for (i = 0; i < s->state.n_entities && (action->item_id == WIZARDRY ||
                                                action->item_id == PRISM); ++i)
            if (!lw_entity_is_dead(s->state.m_entities[i]) &&
                lw_entity_has_effect(s->state.m_entities[i], COVID)) *kept = INFECTED;
    } else if (lw_entity_is_summon(actor)) {
        *kept = SUMMON_ITSELF;
    } else if (why == DAMAGES || why == COSTS) {
        if (holds(actor, MYSTERIOUS_ELECTRISOR) || holds(actor, REVOKED_M_LASER) ||
            holds(actor, PLUTONIUM_BAZOOKA) || holds(actor, UNBRIDLED_GAZOR)) *kept = FED_DAMAGE;
    } else if (why == POISONS) {
        if (holds(actor, ILLICIT_GRENADE_LAUNCHER) || holds(actor, UNBRIDLED_GAZOR))
            *kept = FED_POISON;
    } else if (holds(actor, UNBRIDLED_GAZOR)) {
        *kept = FED_EROSION;
    }
    return 1;
}

/* ---- Useless casts ------------------------------------------------------ */

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

/* Whether two actions lead to the same situation, with the same random numbers. */
static int same_outcome(session *s, const leekrl_action *a, const leekrl_action *b)
{
    static LwState one, two;
    static LwFight first, second;
    int result, same;
    CHECK(lw_fight_step_clone(&first, &one, &s->fight));
    CHECK(lw_fight_step_clone(&second, &two, &s->fight));
    CHECK(leekrl_action_execute(&first, lw_fight_step_current(&first), a, &result) ==
          LEEKRL_ACTION_OK);
    CHECK(leekrl_action_execute(&second, lw_fight_step_current(&second), b, &result) ==
          LEEKRL_ACTION_OK);
    same = same_situation(&one, &two) && one.rng_n == two.rng_n;
    lw_state_dispose(&one);
    lw_state_dispose(&two);
    return same;
}

/* ---- Harmful casts ------------------------------------------------------ */

enum { MEASURES = 15 };
static const char *const measure_names[MEASURES] = {
    "life", "total life", "strength", "wisdom", "agility", "resistance", "science", "magic",
    "frequency", "TP", "MP", "power", "absolute shield", "relative shield", "damage return"
};

static void measure(const LwEntity *e, int *m)
{
    m[0] = lw_entity_get_life(e);
    m[1] = lw_entity_get_total_life(e);
    m[2] = lw_entity_get_strength(e);
    m[3] = lw_entity_get_wisdom(e);
    m[4] = lw_entity_get_agility(e);
    m[5] = lw_entity_get_resistance(e);
    m[6] = lw_entity_get_science(e);
    m[7] = lw_entity_get_magic(e);
    m[8] = lw_entity_get_frequency(e);
    m[9] = lw_entity_get_tp(e);
    m[10] = lw_entity_get_mp(e);
    m[11] = lw_entity_get_power(e);
    m[12] = lw_entity_get_absolute_shield(e);
    m[13] = lw_entity_get_relative_shield(e);
    m[14] = lw_entity_get_damage_return(e);
}

/* What is better for a team with the cast than without it, or NULL. */
static const char *better(const LwState *with, const LwState *without, int team,
                          int only_deaths)
{
    int i, k;
    if (with->n_entities != without->n_entities) return "entities";
    /* When a leek is dead, only who wins counts. */
    for (i = 0; i < with->n_entities; ++i) {
        const LwEntity *x = with->m_entities[i], *y = without->m_entities[i];
        if (lw_entity_is_summon(x) || (!lw_entity_is_dead(x) && !lw_entity_is_dead(y))) continue;
        return lw_entity_is_dead(x) != lw_entity_is_dead(y) &&
               lw_entity_is_dead(x) != (lw_entity_get_team(x) == team) ? "winner" : NULL;
    }
    for (i = 0; i < with->n_entities; ++i) {
        const LwEntity *x = with->m_entities[i], *y = without->m_entities[i];
        int ours = lw_entity_get_team(x) == team, a[MEASURES], b[MEASURES];
        uint32_t lost = y->states & ~x->states, gained = x->states & ~y->states;
        if (lw_entity_is_dead(x) != lw_entity_is_dead(y)) {
            if (lw_entity_is_dead(x) != ours) return "death";
            continue;
        }
        if (lw_entity_is_dead(x) || only_deaths) continue;
        if ((x->cell ? x->cell->id : -1) != (y->cell ? y->cell->id : -1)) return "cell";
        if (lost || (gained & ~(ours ? 1u << LW_ENTITY_STATE_UNHEALABLE : 0u))) return "states";
        measure(x, a);
        measure(y, b);
        for (k = 0; k < MEASURES; ++k)
            if (ours ? a[k] > b[k] : a[k] < b[k]) return measure_names[k];
    }
    return NULL;
}

static int plants(const LwState *state)
{
    int i;
    for (i = 0; i < state->n_entities; ++i)
        if (!lw_entity_is_dead(state->m_entities[i]) && state->m_entities[i]->awakening_zone > 0)
            return 1;
    return 0;
}

/* Follows the fight with and without the cast, until someone dies in only one of
 * them. Returns what the caster's team ever gets from it, or NULL. turns is how
 * many were followed. */
static const char *helps(session *s, LwEntity *actor, const leekrl_action *action, int *turns)
{
    static LwState with, without;
    static LwFight fight, other;
    LwEntity *a = lw_fight_step_current(&s->fight), *b = NULL;
    const char *found;
    int result, parted = 0, team = lw_entity_get_team(actor);
    *turns = 0;
    CHECK(lw_fight_step_clone(&fight, &with, &s->fight));
    CHECK(lw_fight_step_clone(&other, &without, &s->fight));
    CHECK(leekrl_action_execute(&fight, lw_fight_step_current(&fight), action, &result) ==
          LEEKRL_ACTION_OK);
    found = better(&with, &without, team, 0);
    /* Plants act by themselves, differently in each fight. */
    while (!found && !parted && *turns < 40 && !plants(&with)) {
        with.rng_n = without.rng_n;
        if (a) {
            CHECK(!lw_fight_step_end_turn(&fight));
            CHECK(!lw_fight_step_end_turn(&other));
        }
        parted = lw_fight_step_is_over(&fight) || lw_fight_step_is_over(&other);
        if (!parted) {
            CHECK(!lw_fight_step_begin_turn(&fight, &a));
            CHECK(!lw_fight_step_begin_turn(&other, &b));
            parted = !a != !b || (a && lw_entity_get_fid(a) != lw_entity_get_fid(b));
        }
        found = better(&with, &without, team, parted);
        ++*turns;
    }
    lw_state_dispose(&with);
    lw_state_dispose(&without);
    return found;
}

/* ---- Fights ------------------------------------------------------------- */

static const int legendaries[] = {
    0, ILLICIT_GRENADE_LAUNCHER, REVOKED_M_LASER, UNBRIDLED_GAZOR, MYSTERIOUS_ELECTRISOR, 0,
    PLUTONIUM_BAZOOKA, DARK_KATANA, EXPLORER_RIFLE, ENHANCED_LIGHTNINGER, UNSTABLE_DESTROYER
};
static const int hurting[] = {
    SHOCK, PEBBLE, ICE, SPARK, FLASH, DEVIL_STRIKE, PLASMA, TOXIN, PLAGUE, DESINTEGRATION,
    PUNISHMENT, TRANQUILIZER, SOPORIFIC, SLOW_DOWN, BALL_AND_CHAIN, FRACTURE, CRUSHING,
    BRAINWASHING, HEMORRHAGE, COVID, SUPERINFECTION, FIRE_BULB, ICED_BULB, LIGHTNING_BULB,
    WIZARD_BULB, PUNY_BULB, ANTIDOTE, LIBERATION, ALTERATION
};
static const int helping[] = {
    BANDAGE, CURE, DRIP, THERAPY, REGENERATION, ARMORING, ELEVATION, MUTATION, TRANSMUTATION,
    KNOWLEDGE, PRISM, WIZARDRY, MANUMISSION, BRAMBLE, TRANQUILIZER, HEMORRHAGE, FRACTURE,
    PUNY_BULB, WIZARD_BULB, COVID
};

/* Equips what the list is about: random builds rarely hold enough of it. */
static void equip(LwEntityInfo *info, int fight, int team)
{
    const int *chips = fight % 3 == 1 ? hurting : helping;
    int n = fight % 3 == 1 ? (int)(sizeof(hurting) / sizeof(*hurting))
                           : (int)(sizeof(helping) / sizeof(*helping));
    int legendary = legendaries[(size_t)(fight / 3 + 5 * team) %
                                (sizeof(legendaries) / sizeof(*legendaries))];
    info->n_weapons = info->n_chips = 0;
    info->weapons[info->n_weapons++] = GAZOR;
    info->weapons[info->n_weapons++] = J_LASER;
    info->weapons[info->n_weapons++] = QUANTUM_RIFLE;
    if (legendary) info->weapons[info->n_weapons++] = legendary;
    while (info->n_chips < n) {
        info->chips[info->n_chips] = chips[info->n_chips];
        ++info->n_chips;
    }
    info->tp += 20;
    info->mp += 6;
    info->life += 3000;
}

int main(int argc, char **argv)
{
    char error[512];
    int fights = argc > 1 ? atoi(argv[1]) : 33, first = argc > 2 ? atoi(argv[2]) : 0, i;
    session *s = malloc(sizeof(*s));
    leekrl_build_pool *pool;
    leekrl_build_options options;
    const leekrl_action *actions;
    uint8_t *legal, *useful, *mask;
    uint64_t rng = 77 + (uint64_t)first * UINT64_C(1000003);
    size_t n, k;
    long decisions = 0, casts = 0, kept = 0, trimmed = 0, tried = 0, kept_tried = 0, effective = 0;
    long harmful = 0, followed = 0, turns = 0, empty_kept = 0, by_reason[HELPS + 1] = {0};
    long excepted[EXCEPTIONS] = {0}, followed_excepted[EXCEPTIONS] = {0}, helped[EXCEPTIONS] = {0};
    long covid_itself = 0, bramble_enemy = 0, superinfection_own = 0, own_summons = 0, steals = 0;
    long heals = 0, idle_crits = 0, idle[5] = {0}, alike = 0;
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
    mask = malloc(n);
    CHECK(s && pool && legal && useful && mask);
    leekrl_build_options_init(&options);
    for (i = 0; i < fights; ++i) {
        LwEntity *actor;
        int team;
        current_seed = first + i + 1;
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
            if ((first + i) % 3) equip(&info, first + i, team);
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
                CHECK(leekrl_action_catalog_trimmed_mask(catalog, &s->fight, actor,
                          LEEKRL_TRIM_NOTHING, useful) == LEEKRL_ACTION_OK);
                CHECK(leekrl_action_catalog_mask(catalog, &s->fight, actor, legal) ==
                      LEEKRL_ACTION_OK);
                CHECK(!memcmp(legal, useful, n));
                CHECK(leekrl_action_catalog_trimmed_mask(catalog, &s->fight, actor,
                          LEEKRL_TRIM_USELESS, useful) == LEEKRL_ACTION_OK);
                CHECK(leekrl_action_catalog_trimmed_mask(catalog, &s->fight, actor,
                          LEEKRL_TRIM_HARMFUL, mask) == LEEKRL_ACTION_OK);
                ++decisions;
                for (k = 0; k < n; ++k) {
                    const leekrl_action *action = &actions[k];
                    int cast = action->kind >= LEEKRL_ACTION_WEAPON, followed_turns;
                    const char *gain;
                    exception exempt;
                    reached r;
                    CHECK(mask[k] <= useful[k] && useful[k] <= legal[k]);
                    if (!cast) CHECK(mask[k] == legal[k]);
                    if (!legal[k] || !cast) continue;
                    ++casts;
                    kept += mask[k];
                    trimmed += !useful[k];
                    if (!useful[k]) {
                        r.n = 0;
                        if (conditional(action->item_id)) reach(s, actor, action, &r);
                        if (!r.n && next_random(&rng) % 60) continue;
                        ++tried;
                        reach(s, actor, action, &r);
                        steals += !r.n && (action->item_id == J_LASER ||
                                           action->item_id == QUANTUM_RIFLE);
                        heals += r.n && (action->item_id == BANDAGE || action->item_id == CURE ||
                                         action->item_id == REGENERATION);
                        idle_crits += holds(actor, UNSTABLE_DESTROYER);
                        if (r.n) {
                            idle[0] += action->item_id == ALTERATION ||
                                       action->item_id == DESINTEGRATION;
                            idle[1] += action->item_id == SUPERINFECTION;
                            idle[2] += action->item_id == ANTIDOTE;
                            idle[3] += action->item_id == LIBERATION;
                            idle[4] += action->item_id == COVID;
                        }
                        if (changes(s, actor, action)) {
                            size_t j;
                            for (j = 0; j < n; ++j)
                                if (useful[j] && actions[j].kind == action->kind &&
                                    actions[j].item_id == action->item_id &&
                                    same_outcome(s, action, &actions[j])) break;
                            if (j == n) {
                                fprintf(stderr, "seed %d: trimmed cast of item %d at (%d, %d) is "
                                        "not useless\n", current_seed, action->item_id,
                                        action->dx, action->dy);
                                exit(1);
                            }
                            ++alike;
                        }
                        continue;
                    }
                    if (on_the_list(s, actor, action, &exempt) != (!mask[k] || exempt) ||
                        (exempt && !mask[k])) {
                        reach(s, actor, action, &r);
                        fprintf(stderr, "seed %d: item %d at (%d, %d), reaching %d caster, %d "
                                "allies, %d enemies, exception %s: mask %d\n", current_seed,
                                action->item_id, action->dx, action->dy, r.caster, r.allies,
                                r.enemies, exception_names[exempt], mask[k]);
                        exit(1);
                    }
                    if (!mask[k]) {
                        ++harmful;
                        ++by_reason[listed(action->item_id)];
                        if (next_random(&rng) % 25) continue;
                        gain = helps(s, actor, action, &followed_turns);
                        ++followed;
                        turns += followed_turns;
                        if (gain) {
                            fprintf(stderr, "seed %d: masked cast of item %d at (%d, %d) gives: "
                                    "%s, after %d turns\n", current_seed, action->item_id,
                                    action->dx, action->dy, gain, followed_turns);
                            exit(1);
                        }
                    } else if (exempt) {
                        ++excepted[exempt];
                        if (next_random(&rng) % 25) continue;
                        ++followed_excepted[exempt];
                        helped[exempt] += helps(s, actor, action, &followed_turns) != NULL;
                    } else {
                        if (special(action->item_id)) {
                            reach(s, actor, action, &r);
                            covid_itself += action->item_id == COVID && r.caster && r.n == 1;
                            bramble_enemy += action->item_id == BRAMBLE && r.enemies;
                            superinfection_own += action->item_id == SUPERINFECTION &&
                                                  !r.enemies;
                            own_summons += listed(action->item_id) != NOT_LISTED &&
                                           r.allies && !r.caster && !r.enemies;
                        }
                        if (next_random(&rng) % 40) continue;
                        ++kept_tried;
                        effective += changes(s, actor, action);
                        empty_kept += !lw_cell_get_player(
                            lw_map_get_cell_xy(s->state.map, actor->cell->x + action->dx,
                                               actor->cell->y + action->dy), s->state.map);
                    }
                }
                for (k = 0; k < n; ++k) choices += mask[k];
                pick = (size_t)(next_random(&rng) % choices);
                for (k = 0; k < n; ++k)
                    if (mask[k] && pick-- == 0) break;
                if (!k) break;
                CHECK(leekrl_action_execute(&s->fight, actor, &actions[k], &result) ==
                      LEEKRL_ACTION_OK);
            }
            CHECK(!lw_fight_step_end_turn(&s->fight));
        }
        lw_state_dispose(&s->state);
        lw_scenario_dispose(&s->scenario);
    }
    printf("%ld decisions, %ld legal casts: %.1f%% useless, %.1f%% harmful, %.1f%% kept\n",
           decisions, casts, 100.0 * (double)trimmed / (double)casts,
           100.0 * (double)harmful / (double)casts, 100.0 * (double)kept / (double)casts);
    printf("%ld useless casts tried, without effect (%ld steals reaching nobody, %ld heals, %ld "
           "with a weapon healing on critical hits, %ld maximum life taken at full life, %ld "
           "superinfections, %ld antidotes, %ld liberations, %ld covids)\n", tried, steals, heals,
           idle_crits, idle[0], idle[1], idle[2], idle[3], idle[4]);
    printf("%ld trimmed casts did what a kept one of the same item does\n", alike);
    printf("%ld harmful casts followed for %.1f turns, that never helped\n", followed,
           (double)turns / (double)followed);
    printf("harmful: %ld damage the caster, %ld poison it, %ld take its maximum life, %ld only "
           "cost it life, %ld weaken their team, %ld help the enemy\n", by_reason[DAMAGES],
           by_reason[POISONS], by_reason[ERODES], by_reason[COSTS], by_reason[WEAKENS],
           by_reason[HELPS]);
    for (i = 1; i < EXCEPTIONS; ++i)
        printf("exception, %s: %ld casts kept, %ld followed, %ld helped\n", exception_names[i],
               excepted[i], followed_excepted[i], helped[i]);
    printf("%ld kept casts tried, %.0f%% had an effect, %.0f%% on an empty cell; %ld covid on "
           "itself, %ld bramble on an enemy, %ld superinfection on its team, %ld harming "
           "only its summons\n", kept_tried, 100.0 * (double)effective / (double)kept_tried,
           100.0 * (double)empty_kept / (double)kept_tried, covid_itself, bramble_enemy,
           superinfection_own, own_summons);
    CHECK(tried > 1000 && kept_tried > 1000 && trimmed > casts / 2 && followed > 1000);
    CHECK(effective > kept_tried / 3);
    for (i = DAMAGES; i <= HELPS; ++i) CHECK(by_reason[i]);
    for (i = 1; i < EXCEPTIONS; ++i) CHECK(excepted[i]);
    CHECK(alike);
    CHECK(idle[0] && idle[1] && idle[2] && idle[3] && idle[4]);
    CHECK(steals && heals && idle_crits && covid_itself && bramble_enemy && superinfection_own && own_summons);
    CHECK(helped[FED_DAMAGE] && helped[FED_POISON] && helped[FED_EROSION] && helped[REPLACED]);
    CHECK(leekrl_action_catalog_trimmed_mask(catalog, &s->fight, NULL, LEEKRL_TRIM_HARMFUL,
                                             mask) == LEEKRL_ACTION_INVALID_ARGUMENT);
    free(legal);
    free(useful);
    free(mask);
    free(s);
    leekrl_build_pool_destroy(pool);
    leekrl_action_catalog_destroy(catalog);
    leekrl_item_snapshot_clear();
    return 0;
}
