#ifndef LEEKRL_BUILD_H
#define LEEKRL_BUILD_H

#include "leekrl/actions.h"
#include "leekrl/item_snapshot.h"
#include "lw_entity_info.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Legal level 301 leek builds. Needs the item snapshot.
 *
 * Rules: 1780 capital points, all spent; 8 unique components; 4 unique weapons,
 * at most one forgotten (legendary); unique chips, as many as RAM allows and at
 * most 32. Component alterations are not supported.
 */
#define LEEKRL_BUILD_LEVEL 301
#define LEEKRL_BUILD_CAPITAL 1780
#define LEEKRL_BUILD_COMPONENTS 8
#define LEEKRL_BUILD_WEAPONS 4
#define LEEKRL_BUILD_CHIPS 32
#define LEEKRL_BUILD_MIN_LIFE 10

typedef struct {
    int components[LEEKRL_BUILD_COMPONENTS]; /* component ids */
    int bought[LEEKRL_STAT_COUNT];           /* points bought with capital */
    int weapons[LEEKRL_BUILD_WEAPONS];       /* item ids */
    int chips[LEEKRL_BUILD_CHIPS];           /* item ids */
    int n_chips;
} leekrl_build;

/*
 * Relative chance of each stat to receive capital: a stat's share of the
 * capital is drawn as (-log u)^skew * weight, u uniform. A skew of 1 spreads
 * capital evenly on average; more than 1 makes specialized builds likelier. A
 * weight of 0 removes a stat. Life must keep a positive weight.
 *
 * Defaults: skew 2, weight 1, except 0 for cores (they only limit AI operations,
 * which are not simulated) and 0.2 for frequency (mostly turn order).
 */
typedef struct {
    double skew;
    double weights[LEEKRL_STAT_COUNT];
} leekrl_build_options;

void leekrl_build_options_init(leekrl_build_options *options);

/* Items a random build may equip: those of an action catalog. */
typedef struct leekrl_build_pool leekrl_build_pool;

/* NULL if there are too few items or on allocation failure. */
leekrl_build_pool *leekrl_build_pool_create(const leekrl_action_catalog *catalog);
void leekrl_build_pool_destroy(leekrl_build_pool *pool);

/*
 * Draw a legal build. rng is any 64-bit state, advanced by the call: the same
 * state, pool and options always give the same build. Components recovery_core
 * and recovery_ram (legacy) and rgb (irrelevant) are never drawn. Weapons and
 * chips are uniform, chips filling all available RAM up to 32. Returns 0 on
 * invalid arguments, else 1.
 */
int leekrl_build_random(const leekrl_build_pool *pool, const leekrl_build_options *options,
                        uint64_t *rng, leekrl_build *build);

/* Returns 1 if the build follows the rules above and the pool, with at most as
 * many chips as allowed. Else returns 0 and describes the problem. */
int leekrl_build_validate(const leekrl_build_pool *pool, const leekrl_build *build,
                          char *error, size_t error_size);

/* Final value of a stat: base + bought + components. Life is at least 10. Other
 * stats can be negative, as in the game. */
int leekrl_build_stat(const leekrl_build *build, leekrl_stat stat);

/* Value of a stat at level 301, before capital and components. */
int leekrl_build_base(leekrl_stat stat);

/* Capital needed to buy that many points of a stat from zero, or -1 if no
 * purchase sequence gives exactly that amount. */
int leekrl_build_cost(leekrl_stat stat, int bought);

/* Fill a leek definition, released with lw_entity_info_dispose. */
void leekrl_build_to_entity(const leekrl_build *build, int id, LwEntityInfo *info);

#ifdef __cplusplus
}
#endif
#endif
