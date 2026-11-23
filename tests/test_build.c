#include "leekrl/battle.h"
#include "leekrl/build.h"
#include "leekrl/rules.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static leekrl_action_catalog *catalog;
static leekrl_build_pool *pool;

/* Reference values come from the game client's own tests (capital.test.ts) and
 * costs.ts. */
static void test_costs(void)
{
    CHECK(leekrl_build_cost(LEEKRL_STAT_STRENGTH, 0) == 0);
    CHECK(leekrl_build_cost(LEEKRL_STAT_STRENGTH, 2) == 1);
    CHECK(leekrl_build_cost(LEEKRL_STAT_STRENGTH, 1) == -1);
    CHECK(leekrl_build_cost(LEEKRL_STAT_STRENGTH, 200) == 100);
    CHECK(leekrl_build_cost(LEEKRL_STAT_MAGIC, 350) == 250);
    CHECK(leekrl_build_cost(LEEKRL_STAT_WISDOM, 400) == 300);
    CHECK(leekrl_build_cost(LEEKRL_STAT_AGILITY, 600) == 700);
    CHECK(leekrl_build_cost(LEEKRL_STAT_SCIENCE, 601) == 703);
    CHECK(leekrl_build_cost(LEEKRL_STAT_LIFE, 1000) == 250);
    CHECK(leekrl_build_cost(LEEKRL_STAT_LIFE, 1003) == 251);
    /* 334 purchases of 3 cross 2000 and reach 2002. */
    CHECK(leekrl_build_cost(LEEKRL_STAT_LIFE, 2002) == 584);
    CHECK(leekrl_build_cost(LEEKRL_STAT_LIFE, 2000) == -1);
    CHECK(leekrl_build_cost(LEEKRL_STAT_LIFE, 2004) == 585);
    CHECK(leekrl_build_cost(LEEKRL_STAT_FREQUENCY, 1780) == 1780);
    CHECK(leekrl_build_cost(LEEKRL_STAT_FREQUENCY, 1781) == -1);
    CHECK(leekrl_build_cost(LEEKRL_STAT_CORES, 9) == 540);
    CHECK(leekrl_build_cost(LEEKRL_STAT_RAM, 10) == 640);
    CHECK(leekrl_build_cost(LEEKRL_STAT_TP, 15) == 975);
    CHECK(leekrl_build_cost(LEEKRL_STAT_TP, 16) == 1075);
    CHECK(leekrl_build_cost(LEEKRL_STAT_MP, 9) == 900);
    CHECK(leekrl_build_cost(LEEKRL_STAT_MP, 10) == 1080);
    CHECK(leekrl_build_cost(LEEKRL_STAT_COUNT, 0) == -1);
    CHECK(leekrl_build_cost(LEEKRL_STAT_LIFE, -1) == -1);
}

static int component(const char *name)
{
    size_t i, n;
    const leekrl_component *list = leekrl_components(&n);
    for (i = 0; i < n; ++i)
        if (!strcmp(list[i].name, name)) return list[i].id;
    CHECK(0);
    return 0;
}

static void test_random(void)
{
    leekrl_build_options options;
    leekrl_build build, again;
    uint64_t rng, copy;
    char error[128];
    int i, j, capped = 0, floor_life = 0, specialized = 0;
    int low[LEEKRL_STAT_COUNT], high[LEEKRL_STAT_COUNT];
    double mean[LEEKRL_STAT_COUNT] = {0};
    static const char *const names[] = {"life", "strength", "wisdom", "agility", "resistance",
        "science", "magic", "frequency", "cores", "ram", "tp", "mp"};
    const int n = 20000;
    int legacy[3], seen_weapon[1024] = {0}, seen_chip[1024] = {0}, weapons = 0, chips = 0;
    legacy[0] = component("recovery_core");
    legacy[1] = component("recovery_ram");
    legacy[2] = component("rgb");
    leekrl_build_options_init(&options);
    for (j = 0; j < LEEKRL_STAT_COUNT; ++j) {
        low[j] = 1 << 30;
        high[j] = -(1 << 30);
    }
    rng = 42;
    for (i = 0; i < n; ++i) {
        LwEntityInfo info;
        int ram, top = 0;
        copy = rng;
        CHECK(leekrl_build_random(pool, &options, &rng, &build));
        CHECK(leekrl_build_random(pool, &options, &copy, &again));
        CHECK(copy == rng && !memcmp(&build, &again, sizeof(build)));
        if (!leekrl_build_validate(pool, &build, error, sizeof(error))) {
            fprintf(stderr, "build %d: %s\n", i, error);
            exit(1);
        }
        for (j = 0; j < LEEKRL_BUILD_COMPONENTS; ++j)
            CHECK(build.components[j] != legacy[0] && build.components[j] != legacy[1] &&
                  build.components[j] != legacy[2]);
        ram = leekrl_build_stat(&build, LEEKRL_STAT_RAM);
        CHECK(!build.bought[LEEKRL_STAT_CORES]);
        CHECK(ram >= 6 && build.n_chips == (ram < 32 ? ram : 32));
        capped += ram > 32;
        CHECK(leekrl_build_stat(&build, LEEKRL_STAT_LIFE) >= 10);
        floor_life += leekrl_build_stat(&build, LEEKRL_STAT_LIFE) == 10;
        for (j = 0; j < LEEKRL_STAT_COUNT; ++j) {
            int value = leekrl_build_stat(&build, (leekrl_stat)j);
            int cost = leekrl_build_cost((leekrl_stat)j, build.bought[j]);
            if (value < low[j]) low[j] = value;
            if (value > high[j]) high[j] = value;
            mean[j] += value;
            if (cost > top) top = cost;
        }
        specialized += top > LEEKRL_BUILD_CAPITAL / 2;
        for (j = 0; j < LEEKRL_BUILD_WEAPONS; ++j) {
            CHECK(!leekrl_solo_item_excluded(build.weapons[j]));
            weapons += !seen_weapon[build.weapons[j]]++;
        }
        for (j = 0; j < build.n_chips; ++j) {
            CHECK(!leekrl_solo_item_excluded(build.chips[j]));
            chips += !seen_chip[build.chips[j]]++;
        }
        leekrl_build_to_entity(&build, 5, &info);
        CHECK(info.id == 5 && info.level == 301 && info.n_weapons == 4 &&
              info.n_chips == build.n_chips && info.ram == ram);
        lw_entity_info_dispose(&info);
    }
    CHECK(weapons == 37 && chips == 105);
    printf("%d builds: %d with RAM over 32, %d at the life floor, %d with over half of the "
           "capital in one stat\n", n, capped, floor_life, specialized);
    for (j = 0; j < LEEKRL_STAT_COUNT; ++j)
        printf("  %-10s min %5d  mean %7.1f  max %5d\n", names[j], low[j], mean[j] / n, high[j]);

    /* Unwanted stats get nothing. */
    options.weights[LEEKRL_STAT_FREQUENCY] = 0;
    for (i = 0; i < 1000; ++i) {
        CHECK(leekrl_build_random(pool, &options, &rng, &build));
        CHECK(!build.bought[LEEKRL_STAT_CORES] && !build.bought[LEEKRL_STAT_FREQUENCY]);
        CHECK(leekrl_build_validate(pool, &build, error, sizeof(error)));
    }
    /* Everything in one stat. */
    for (j = 1; j < LEEKRL_STAT_COUNT; ++j) options.weights[j] = 0;
    CHECK(leekrl_build_random(pool, &options, &rng, &build));
    CHECK(leekrl_build_cost(LEEKRL_STAT_LIFE, build.bought[LEEKRL_STAT_LIFE]) == 1780);
    options.weights[LEEKRL_STAT_LIFE] = 0;
    CHECK(!leekrl_build_random(pool, &options, &rng, &build));
}

static void test_validation(void)
{
    leekrl_build_options options;
    leekrl_build good, bad;
    uint64_t rng = 7;
    char error[128];
    int plants = 444;
    leekrl_build_options_init(&options);
    do CHECK(leekrl_build_random(pool, &options, &rng, &good));
    while (good.n_chips < 10 || good.n_chips > 20);
#define REJECTED(change, text) do { \
        bad = good; \
        change; \
        CHECK(!leekrl_build_validate(pool, &bad, error, sizeof(error))); \
        CHECK(strstr(error, text)); \
    } while (0)
    REJECTED(bad.components[1] = bad.components[0], "duplicate component");
    REJECTED(bad.components[1] = 9999, "unknown component");
    REJECTED(bad.bought[LEEKRL_STAT_FREQUENCY] += 1, "capital");
    REJECTED(bad.bought[LEEKRL_STAT_LIFE] += 1, "impossible purchase");
    REJECTED(bad.weapons[1] = bad.weapons[0], "duplicate weapon");
    REJECTED(bad.weapons[1] = 410, "weapon not allowed");
    REJECTED(bad.chips[1] = bad.chips[0], "duplicate chip");
    REJECTED(bad.chips[1] = plants, "chip not allowed");
    REJECTED(bad.chips[1] = 84, "chip not allowed");
    REJECTED(bad.n_chips = 33, "number of chips");
    REJECTED(bad.n_chips = leekrl_build_stat(&good, LEEKRL_STAT_RAM) + 1, "number of chips");
    REJECTED(bad.weapons[0] = 187; bad.weapons[1] = 175, "forgotten");
    bad = good;
    bad.n_chips = 0; /* fewer chips than RAM is legal */
    CHECK(leekrl_build_validate(pool, &bad, error, sizeof(error)));

    /* The life floor: -400 -300 -200 -100 -100 from components, 1000 base. */
    memset(&bad, 0, sizeof(bad));
    bad.components[0] = component("pear");
    bad.components[1] = component("switch2");
    bad.components[2] = component("nuclear_core");
    bad.components[3] = component("switch");
    bad.components[4] = component("cocoa");
    CHECK(leekrl_build_stat(&bad, LEEKRL_STAT_LIFE) == 10);
    bad.bought[LEEKRL_STAT_LIFE] = 112;
    CHECK(leekrl_build_stat(&bad, LEEKRL_STAT_LIFE) == 12);
    /* Other stats can be negative: -40 -40 -30 resistance. */
    memset(&bad, 0, sizeof(bad));
    bad.components[0] = component("levitator");
    bad.components[1] = component("propulsor2");
    bad.components[2] = component("propulsor");
    CHECK(leekrl_build_stat(&bad, LEEKRL_STAT_RESISTANCE) == -110);
    bad.bought[LEEKRL_STAT_RESISTANCE] = 120;
    CHECK(leekrl_build_stat(&bad, LEEKRL_STAT_RESISTANCE) == 10);
}

/* Random builds must be accepted by the battle environment and playable. */
static void test_battles(void)
{
    leekrl_build_options options;
    uint64_t rng = 99;
    int i, steps = 0;
    leekrl_build_options_init(&options);
    for (i = 0; i < 20; ++i) {
        LwEntityInfo info[2];
        leekrl_battle_options battle_options;
        leekrl_battle *battle = NULL;
        leekrl_env *env = NULL;
        leekrl_build build;
        int j;
        size_t n;
        for (j = 0; j < 2; ++j) {
            CHECK(leekrl_build_random(pool, &options, &rng, &build));
            leekrl_build_to_entity(&build, j + 1, &info[j]);
        }
        leekrl_battle_options_init(&battle_options);
        CHECK(leekrl_battle_create(&info[0], &info[1], &battle_options, &battle) ==
              LEEKRL_ENV_OK);
        n = leekrl_action_catalog_count(leekrl_battle_catalog(battle));
        CHECK(n == leekrl_action_catalog_count(catalog));
        CHECK(leekrl_env_create(leekrl_battle_observation_size(battle), n,
                                leekrl_battle_callbacks(), battle, &env) == LEEKRL_ENV_OK);
        CHECK(leekrl_env_reset(env, (uint64_t)i) == LEEKRL_ENV_OK);
        for (j = 0; j < 500; ++j) {
            const uint8_t *mask = leekrl_env_action_mask(env);
            leekrl_step_result result;
            size_t k, legal = 0, pick;
            for (k = 0; k < n; ++k) legal += mask[k];
            rng = rng * UINT64_C(6364136223846793005) + 1;
            pick = (size_t)((rng >> 33) % legal);
            for (k = 0; k < n; ++k)
                if (mask[k] && pick-- == 0) break;
            CHECK(leekrl_env_step(env, k, &result) == LEEKRL_ENV_OK);
            ++steps;
            if (result.terminated || result.truncated) break;
        }
        leekrl_env_destroy(env);
        leekrl_battle_destroy(battle);
    }
    printf("%d battle steps\n", steps);
}

int main(void)
{
    char error[512];
    if (!leekrl_item_snapshot_register(error, sizeof(error))) {
        fprintf(stderr, "item registration: %s\n", error);
        return 1;
    }
    CHECK(leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
                                               leekrl_solo_excluded_count, &catalog) ==
          LEEKRL_ACTION_OK);
    pool = leekrl_build_pool_create(catalog);
    CHECK(pool);
    test_costs();
    test_random();
    test_validation();
    test_battles();
    leekrl_build_pool_destroy(pool);
    leekrl_action_catalog_destroy(catalog);
    leekrl_item_snapshot_clear();
    return 0;
}
