#include "leekrl/build.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lw_constants.h"

/* From `from` points bought, `capital` buys `gain` points. The last step of a
 * stat has no upper bound. Mirrors COSTS in the game client's costs.ts. */
typedef struct {
    int from;
    int capital;
    int gain;
} cost_step;

#define MAX_STEPS 15

static const struct {
    int count;
    cost_step steps[MAX_STEPS];
} costs[LEEKRL_STAT_COUNT] = {
#define MAIN_STAT {4, {{0, 1, 2}, {200, 1, 1}, {400, 2, 1}, {600, 3, 1}}}
#define CORE_STAT {9, {{0, 20, 1}, {1, 30, 1}, {2, 40, 1}, {3, 50, 1}, {4, 60, 1}, \
                       {5, 70, 1}, {6, 80, 1}, {7, 90, 1}, {8, 100, 1}}}
    {3, {{0, 1, 4}, {1000, 1, 3}, {2000, 1, 2}}}, /* life */
    MAIN_STAT,                                    /* strength */
    MAIN_STAT,                                    /* wisdom */
    MAIN_STAT,                                    /* agility */
    MAIN_STAT,                                    /* resistance */
    MAIN_STAT,                                    /* science */
    MAIN_STAT,                                    /* magic */
    {1, {{0, 1, 1}}},                             /* frequency */
    CORE_STAT,                                    /* cores */
    CORE_STAT,                                    /* ram */
    {15, {{0, 30, 1}, {1, 35, 1}, {2, 40, 1}, {3, 45, 1}, {4, 50, 1}, {5, 55, 1},
          {6, 60, 1}, {7, 65, 1}, {8, 70, 1}, {9, 75, 1}, {10, 80, 1}, {11, 85, 1},
          {12, 90, 1}, {13, 95, 1}, {14, 100, 1}}}, /* tp */
    {9, {{0, 20, 1}, {1, 40, 1}, {2, 60, 1}, {3, 80, 1}, {4, 100, 1}, {5, 120, 1},
         {6, 140, 1}, {7, 160, 1}, {8, 180, 1}}}    /* mp */
#undef MAIN_STAT
#undef CORE_STAT
};

/* Level 301, before capital and components. */
static const int base_stats[LEEKRL_STAT_COUNT] = {
    1000, 0, 0, 0, 0, 0, 0, 100, 1, 6, 10, 3
};

/* Never drawn: legacy components, and rgb which only gives 1 life. */
static const char *const undrawn[] = {"recovery_core", "recovery_ram", "rgb"};

struct leekrl_build_pool {
    int *weapons;
    int *chips;
    int n_weapons;
    int n_chips;
};

static const cost_step *next_step(int stat, int bought)
{
    int i = costs[stat].count - 1;
    while (i > 0 && costs[stat].steps[i].from > bought) --i;
    return &costs[stat].steps[i];
}

int leekrl_build_base(leekrl_stat stat)
{
    return (int)stat >= 0 && stat < LEEKRL_STAT_COUNT ? base_stats[stat] : 0;
}

int leekrl_build_cost(leekrl_stat stat, int bought)
{
    int total = 0, points = 0;
    if ((int)stat < 0 || stat >= LEEKRL_STAT_COUNT || bought < 0) return -1;
    while (points < bought) {
        const cost_step *step = next_step(stat, points);
        points += step->gain;
        total += step->capital;
        if (total > LEEKRL_BUILD_CAPITAL) return -1;
    }
    return points == bought ? total : -1;
}

static const leekrl_component *find_component(int id)
{
    size_t i, n;
    const leekrl_component *list = leekrl_components(&n);
    for (i = 0; i < n; ++i)
        if (list[i].id == id) return &list[i];
    return NULL;
}

int leekrl_build_stat(const leekrl_build *build, leekrl_stat stat)
{
    int i, value = base_stats[stat] + build->bought[stat] + build->altered[stat];
    for (i = 0; i < LEEKRL_BUILD_COMPONENTS; ++i) {
        const leekrl_component *component = find_component(build->components[i]);
        if (component) value += component->stats[stat];
    }
    if (stat == LEEKRL_STAT_LIFE && value < LEEKRL_BUILD_MIN_LIFE) return LEEKRL_BUILD_MIN_LIFE;
    return value;
}

static int chip_limit(const leekrl_build *build)
{
    int ram = leekrl_build_stat(build, LEEKRL_STAT_RAM);
    if (ram < 0) return 0;
    return ram < LEEKRL_BUILD_CHIPS ? ram : LEEKRL_BUILD_CHIPS;
}

static int contains(const int *list, int n, int value)
{
    int i;
    for (i = 0; i < n; ++i)
        if (list[i] == value) return 1;
    return 0;
}

/* ---- Randomness (splitmix64) --------------------------------------- */

static uint64_t draw(uint64_t *rng)
{
    uint64_t z = (*rng += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/* In (0, 1]. */
static double draw_unit(uint64_t *rng)
{
    return (double)((draw(rng) >> 11) + 1) / 9007199254740992.0;
}

/* In [0, n). The modulo bias is negligible for these sizes. */
static int draw_below(uint64_t *rng, int n)
{
    return (int)(draw(rng) % (uint64_t)n);
}

/* ---- Pool ----------------------------------------------------------- */

leekrl_build_pool *leekrl_build_pool_create(const leekrl_action_catalog *catalog)
{
    const leekrl_action *actions;
    leekrl_build_pool *pool;
    size_t i, n;
    int regular = 0;
    if (!catalog) return NULL;
    actions = leekrl_action_catalog_actions(catalog);
    n = leekrl_action_catalog_count(catalog);
    pool = calloc(1, sizeof(*pool));
    if (!pool) return NULL;
    /* An upper bound: every item has at least one action. */
    pool->weapons = malloc(n * sizeof(*pool->weapons));
    pool->chips = malloc(n * sizeof(*pool->chips));
    if (!pool->weapons || !pool->chips) {
        leekrl_build_pool_destroy(pool);
        return NULL;
    }
    for (i = 0; i < n; ++i) {
        /* Actions of an item are contiguous. */
        if (i && actions[i].kind == actions[i - 1].kind &&
            actions[i].item_id == actions[i - 1].item_id) continue;
        if (actions[i].kind == LEEKRL_ACTION_WEAPON) {
            pool->weapons[pool->n_weapons++] = actions[i].item_id;
            regular += !leekrl_weapon_forgotten(actions[i].item_id);
        } else if (actions[i].kind == LEEKRL_ACTION_CHIP) {
            pool->chips[pool->n_chips++] = actions[i].item_id;
        }
    }
    if (regular < LEEKRL_BUILD_WEAPONS || pool->n_chips < LEEKRL_BUILD_CHIPS) {
        leekrl_build_pool_destroy(pool);
        return NULL;
    }
    return pool;
}

void leekrl_build_pool_destroy(leekrl_build_pool *pool)
{
    if (!pool) return;
    free(pool->weapons);
    free(pool->chips);
    free(pool);
}

/* ---- Random builds --------------------------------------------------- */

void leekrl_build_options_init(leekrl_build_options *options)
{
    int i;
    if (!options) return;
    options->skew = 2;
    for (i = 0; i < LEEKRL_STAT_COUNT; ++i) options->weights[i] = 1;
    options->weights[LEEKRL_STAT_CORES] = 0;
    options->weights[LEEKRL_STAT_FREQUENCY] = 0.2;
}

static int valid_options(const leekrl_build_options *options)
{
    int i;
    if (!options || !isfinite(options->skew) || options->skew < 0 || options->skew > 10)
        return 0;
    for (i = 0; i < LEEKRL_STAT_COUNT; ++i)
        if (!isfinite(options->weights[i]) || options->weights[i] < 0 ||
            options->weights[i] > 1e6) return 0;
    return options->weights[LEEKRL_STAT_LIFE] > 0;
}

static void spend_capital(const leekrl_build_options *options, uint64_t *rng,
                          leekrl_build *build)
{
    double share[LEEKRL_STAT_COUNT], sum = 0;
    int remaining = LEEKRL_BUILD_CAPITAL, stat;
    for (stat = 0; stat < LEEKRL_STAT_COUNT; ++stat) {
        /* The small constant keeps every share of a wanted stat positive. */
        share[stat] = options->weights[stat] *
                      (pow(-log(draw_unit(rng)), options->skew) + 1e-9);
        sum += share[stat];
    }
    for (stat = 0; stat < LEEKRL_STAT_COUNT; ++stat) {
        int budget = (int)(LEEKRL_BUILD_CAPITAL * (share[stat] / sum));
        const cost_step *step;
        while ((step = next_step(stat, build->bought[stat]))->capital <= budget &&
               step->capital <= remaining) {
            build->bought[stat] += step->gain;
            budget -= step->capital;
            remaining -= step->capital;
        }
    }
    /* Leftovers: one purchase at a time, among the affordable stats. Life
     * always is, as a purchase costs 1. */
    while (remaining > 0) {
        double pick;
        int last = -1;
        sum = 0;
        for (stat = 0; stat < LEEKRL_STAT_COUNT; ++stat)
            if (next_step(stat, build->bought[stat])->capital <= remaining) sum += share[stat];
        pick = draw_unit(rng) * sum;
        for (stat = 0; stat < LEEKRL_STAT_COUNT; ++stat) {
            if (share[stat] <= 0 ||
                next_step(stat, build->bought[stat])->capital > remaining) continue;
            last = stat;
            pick -= share[stat];
            if (pick <= 0) break;
        }
        remaining -= next_step(last, build->bought[last])->capital;
        build->bought[last] += next_step(last, build->bought[last])->gain;
    }
}

static int drawable(const leekrl_component *component)
{
    size_t i;
    for (i = 0; i < sizeof(undrawn) / sizeof(undrawn[0]); ++i)
        if (!strcmp(component->name, undrawn[i])) return 0;
    return 1;
}

int leekrl_build_random(const leekrl_build_pool *pool, const leekrl_build_options *options,
                        uint64_t *rng, leekrl_build *build)
{
    const leekrl_component *components;
    size_t n_components, i;
    int n = 0, forgotten = 0, limit, available = 0;
    if (!pool || !rng || !build || !valid_options(options)) return 0;
    components = leekrl_components(&n_components);
    for (i = 0; i < n_components; ++i) available += drawable(&components[i]);
    if (available < LEEKRL_BUILD_COMPONENTS) return 0;
    memset(build, 0, sizeof(*build));
    while (n < LEEKRL_BUILD_COMPONENTS) {
        const leekrl_component *component =
            &components[draw_below(rng, (int)n_components)];
        if (drawable(component) && !contains(build->components, n, component->id))
            build->components[n++] = component->id;
    }
    spend_capital(options, rng, build);
    for (n = 0; n < LEEKRL_BUILD_WEAPONS;) {
        int id = pool->weapons[draw_below(rng, pool->n_weapons)];
        int legendary = leekrl_weapon_forgotten(id);
        if (contains(build->weapons, n, id) || (legendary && forgotten)) continue;
        forgotten |= legendary;
        build->weapons[n++] = id;
    }
    limit = chip_limit(build);
    while (build->n_chips < limit) {
        int id = pool->chips[draw_below(rng, pool->n_chips)];
        if (!contains(build->chips, build->n_chips, id)) build->chips[build->n_chips++] = id;
    }
    return 1;
}

/* ---- Validation and conversion --------------------------------------- */

static int invalid(char *error, size_t size, const char *message, int value)
{
    if (error && size) snprintf(error, size, "%s (%d)", message, value);
    return 0;
}

int leekrl_build_validate(const leekrl_build_pool *pool, const leekrl_build *build,
                          char *error, size_t error_size)
{
    int i, total = 0, forgotten = 0;
    if (error && error_size) error[0] = 0;
    if (!pool || !build) return invalid(error, error_size, "missing argument", 0);
    for (i = 0; i < LEEKRL_BUILD_COMPONENTS; ++i) {
        if (!find_component(build->components[i]))
            return invalid(error, error_size, "unknown component", build->components[i]);
        if (contains(build->components, i, build->components[i]))
            return invalid(error, error_size, "duplicate component", build->components[i]);
    }
    for (i = 0; i < LEEKRL_STAT_COUNT; ++i) {
        int cost = leekrl_build_cost((leekrl_stat)i, build->bought[i]);
        if (cost < 0) return invalid(error, error_size, "impossible purchase of stat", i);
        total += cost;
    }
    if (total != LEEKRL_BUILD_CAPITAL)
        return invalid(error, error_size, "wrong amount of capital spent", total);
    for (i = 0; i < LEEKRL_BUILD_WEAPONS; ++i) {
        int id = build->weapons[i];
        if (!contains(pool->weapons, pool->n_weapons, id))
            return invalid(error, error_size, "weapon not allowed", id);
        if (contains(build->weapons, i, id))
            return invalid(error, error_size, "duplicate weapon", id);
        forgotten += leekrl_weapon_forgotten(id);
    }
    if (forgotten > 1) return invalid(error, error_size, "too many forgotten weapons", forgotten);
    if (build->n_chips < 0 || build->n_chips > chip_limit(build))
        return invalid(error, error_size, "wrong number of chips", build->n_chips);
    for (i = 0; i < build->n_chips; ++i) {
        int id = build->chips[i];
        if (!contains(pool->chips, pool->n_chips, id))
            return invalid(error, error_size, "chip not allowed", id);
        if (contains(build->chips, i, id))
            return invalid(error, error_size, "duplicate chip", id);
    }
    return 1;
}

void leekrl_build_to_entity(const leekrl_build *build, int id, LwEntityInfo *info)
{
    int i;
    lw_entity_info_init(info);
    info->id = id;
    info->type = LW_ENTITY_TYPE_LEEK;
    info->level = LEEKRL_BUILD_LEVEL;
    info->life = leekrl_build_stat(build, LEEKRL_STAT_LIFE);
    info->strength = leekrl_build_stat(build, LEEKRL_STAT_STRENGTH);
    info->wisdom = leekrl_build_stat(build, LEEKRL_STAT_WISDOM);
    info->agility = leekrl_build_stat(build, LEEKRL_STAT_AGILITY);
    info->resistance = leekrl_build_stat(build, LEEKRL_STAT_RESISTANCE);
    info->science = leekrl_build_stat(build, LEEKRL_STAT_SCIENCE);
    info->magic = leekrl_build_stat(build, LEEKRL_STAT_MAGIC);
    info->frequency = leekrl_build_stat(build, LEEKRL_STAT_FREQUENCY);
    info->cores = leekrl_build_stat(build, LEEKRL_STAT_CORES);
    info->ram = leekrl_build_stat(build, LEEKRL_STAT_RAM);
    info->tp = leekrl_build_stat(build, LEEKRL_STAT_TP);
    info->mp = leekrl_build_stat(build, LEEKRL_STAT_MP);
    for (i = 0; i < LEEKRL_BUILD_WEAPONS; ++i) info->weapons[info->n_weapons++] = build->weapons[i];
    for (i = 0; i < build->n_chips; ++i) info->chips[info->n_chips++] = build->chips[i];
}
