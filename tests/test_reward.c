#include "leekrl/actions.h"
#include "leekrl/battle.h"
#include "leekrl/build.h"
#include "leekrl/item_snapshot.h"
#include "leekrl/plant_ai.h"
#include "leekrl/reward.h"
#include "leekrl/rules.h"
#include "lw_chip.h"
#include "lw_effect.h"
#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_generator.h"
#include "lw_outcome.h"
#include "lw_scenario.h"
#include "lw_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int current_seed;

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: seed %d: %s\n", __FILE__, __LINE__, current_seed, #x); exit(1); \
} } while (0)

#define NEAR(a, b) (fabs((double)(a) - (double)(b)) < 1e-5)

typedef struct {
    LwGenerator generator;
    LwScenario scenario;
    LwState state;
    LwFight fight;
    LwOutcome outcome;
    leekrl_reward_tracker tracker;
    float parts[LEEKRL_REWARD_PARTS];
} session;

static leekrl_reward_options options;
static leekrl_action_catalog *catalog;

static uint64_t next_random(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static void start(session *s, const LwEntityInfo *a, const LwEntityInfo *b, int seed)
{
    memset(s, 0, sizeof(*s));
    lw_generator_init(&s->generator);
    lw_generator_set_plant_ai_dispatch(&s->generator, leekrl_plant_ai, NULL);
    lw_scenario_init(&s->scenario);
    s->scenario.seed = seed;
    CHECK(lw_scenario_add_entity(&s->scenario, 0, a) && lw_scenario_add_entity(&s->scenario, 1, b));
    CHECK(!lw_generator_prepare_scenario(&s->generator, &s->scenario, NULL, NULL, NULL,
                                         &s->state, &s->fight, &s->outcome));
    CHECK(!lw_fight_step_start(&s->fight));
    leekrl_reward_start(&s->tracker, &options, &s->fight);
}

static void stop(session *s)
{
    lw_state_dispose(&s->state);
    lw_scenario_dispose(&s->scenario);
}

static float step(session *s, int team)
{
    leekrl_reward_step(&s->tracker, &options, &s->fight, team, s->parts);
    return leekrl_reward_sum(s->parts);
}

static session *duel(void)
{
    session *s = malloc(sizeof(*s));
    LwEntityInfo info[2];
    LwEntity *actor;
    int i;
    CHECK(s);
    for (i = 0; i < 2; ++i) {
        lw_entity_info_init(&info[i]);
        info[i].id = i + 1;
        info[i].level = 301;
        info[i].life = 2000 + 2000 * i; /* different lives weigh the same */
        info[i].tp = 20;
        info[i].mp = 5;
        info[i].ram = 6;
        info[i].frequency = i ? 100 : 5000;
        info[i].chips[0] = 73;
        info[i].n_chips = 1;
    }
    start(s, &info[0], &info[1], 4);
    CHECK(!lw_fight_step_begin_turn(&s->fight, &actor) && actor && actor->team == 0);
    return s;
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

static void effect(session *s, int id, int turns, double value, LwEntity *target, LwEntity *caster)
{
    lw_effect_create_effect(&s->state, id, turns, 1, value, 0, 0, target, caster, NULL, 0, 1, 0,
                            1, 0, 0);
}

/* The table of damage types, healing, and who is concerned. */
static void test_life(void)
{
    session *s = duel();
    LwEntity *me = leek(s, 0), *enemy = leek(s, 1);
    int life, total;
    float reward;
    CHECK(step(s, 0) == 0 && step(s, 1) == 0);

    /* Direct damage, with erosion: life and total life, each in its part. */
    effect(s, LW_EFFECT_TYPE_DAMAGE, 0, 400, enemy, me);
    CHECK(enemy->life == 3600 && enemy->m_total_life < 4000);
    total = 4000 - enemy->m_total_life;
    reward = step(s, 0);
    CHECK(NEAR(s->parts[LEEKRL_REWARD_LIFE], 0.3 * 400 / 4000));
    CHECK(NEAR(s->parts[LEEKRL_REWARD_TOTAL_LIFE], 0.1 * total / 4000));
    CHECK(NEAR(reward, 0.3 * 400 / 4000 + 0.1 * total / 4000));
    CHECK(NEAR(step(s, 1), -reward));
    CHECK(step(s, 0) == 0); /* nothing happened since */

    /* The same share of a smaller leek weighs the same. */
    effect(s, LW_EFFECT_TYPE_DAMAGE, 0, 200, me, enemy);
    step(s, 0);
    CHECK(NEAR(s->parts[LEEKRL_REWARD_LIFE], -0.3 * 200 / 2000));
    step(s, 1);

    /* Damage then an equal heal: no life reward in the end. */
    life = enemy->life;
    effect(s, LW_EFFECT_TYPE_DAMAGE, 0, 300, enemy, me);
    step(s, 0);
    reward = s->parts[LEEKRL_REWARD_LIFE];
    CHECK(reward > 0);
    effect(s, LW_EFFECT_TYPE_HEAL, 0, 300, enemy, enemy);
    CHECK(enemy->life == life);
    step(s, 0);
    CHECK(NEAR(s->parts[LEEKRL_REWARD_LIFE], -reward));
    step(s, 1);

    /* Nova reduces total life only. */
    life = enemy->life;
    total = enemy->m_total_life;
    effect(s, LW_EFFECT_TYPE_NOVA_DAMAGE, 0, 100, enemy, me);
    CHECK(enemy->life == life && enemy->m_total_life == total - 100);
    step(s, 0);
    CHECK(s->parts[LEEKRL_REWARD_LIFE] == 0);
    CHECK(NEAR(s->parts[LEEKRL_REWARD_TOTAL_LIFE], 0.1 * 100 / 4000));

    /* Poison and aftereffect: the latter at once, the former when it ticks. */
    life = enemy->life;
    effect(s, LW_EFFECT_TYPE_POISON, 2, 150, enemy, me);
    CHECK(step(s, 0) == 0 && enemy->life == life);
    lw_entity_start_turn(enemy);
    CHECK(enemy->life == life - 150);
    step(s, 0);
    CHECK(NEAR(s->parts[LEEKRL_REWARD_LIFE], 0.3 * 150 / 4000));
    effect(s, LW_EFFECT_TYPE_AFTEREFFECT, 2, 80, enemy, me);
    step(s, 0);
    CHECK(NEAR(s->parts[LEEKRL_REWARD_LIFE], 0.3 * 80 / 4000));

    /* A share of the caster's life, and damage sent back to the attacker. */
    life = enemy->life;
    effect(s, LW_EFFECT_TYPE_LIFE_DAMAGE, 0, 10, enemy, me);
    CHECK(enemy->life == life - me->life / 10);
    step(s, 0);
    CHECK(NEAR(s->parts[LEEKRL_REWARD_LIFE], 0.3 * (me->life / 10) / 4000));
    effect(s, LW_EFFECT_TYPE_DAMAGE_RETURN, 3, 50, enemy, enemy);
    life = me->life;
    effect(s, LW_EFFECT_TYPE_DAMAGE, 0, 100, enemy, me);
    CHECK(me->life == life - 50);
    step(s, 0);
    CHECK(NEAR(s->parts[LEEKRL_REWARD_LIFE], 0.3 * 100 / 4000 - 0.3 * 50 / 2000));

    /* Hurting oneself. */
    effect(s, LW_EFFECT_TYPE_DAMAGE, 0, 100, me, me);
    CHECK(step(s, 0) < 0);
    stop(s);
    free(s);
}

/* Summoning then losing the summon gives zero, whoever kills it. */
static void test_summons(void)
{
    session *s = duel();
    LwEntity *me = leek(s, 0), *bulb;
    leekrl_action summon = {LEEKRL_ACTION_CHIP, 73, 0, 0};
    uint8_t legal = 0;
    int result, i;
    float gained, lost;
    for (i = 0; i < 4 && !legal; ++i) {
        summon.dx = i == 0 ? 1 : i == 1 ? -1 : 0;
        summon.dy = i == 2 ? 1 : i == 3 ? -1 : 0;
        CHECK(leekrl_action_mask(&s->fight, me, &summon, 1, &legal) == LEEKRL_ACTION_OK);
    }
    CHECK(legal);
    CHECK(leekrl_action_execute(&s->fight, me, &summon, &result) == LEEKRL_ACTION_OK);
    bulb = lw_state_get_last_entity(&s->state);
    gained = step(s, 0);
    CHECK(NEAR(gained, 0.02) && NEAR(s->parts[LEEKRL_REWARD_SUMMONS], 0.02));
    CHECK(NEAR(step(s, 1), -0.02));
    effect(s, LW_EFFECT_TYPE_DAMAGE, 0, bulb->life / 2, bulb, leek(s, 1));
    lost = step(s, 0);
    CHECK(lost < 0 && lost > -0.02f);
    effect(s, LW_EFFECT_TYPE_DAMAGE, 0, 100000, bulb, leek(s, 1));
    CHECK(lw_entity_is_dead(bulb));
    lost += step(s, 0);
    CHECK(NEAR(gained + lost, 0));
    CHECK(NEAR(step(s, 1), 0.02));
    stop(s);
    free(s);
}

static void test_end(void)
{
    session *s;
    float parts[LEEKRL_REWARD_PARTS];
    int i;
    /* Win on turn 1 with 3 actions. */
    s = duel();
    for (i = 0; i < 3; ++i) leekrl_reward_count_action(&s->tracker, 0);
    leekrl_reward_count_action(&s->tracker, 1);
    effect(s, LW_EFFECT_TYPE_DAMAGE, 0, 100000, leek(s, 1), leek(s, 0));
    CHECK(lw_state_is_finished(&s->state));
    CHECK(!lw_fight_step_end_turn(&s->fight) && lw_fight_step_is_over(&s->fight));
    step(s, 0);
    CHECK(NEAR(s->parts[LEEKRL_REWARD_LIFE], 0.3) && NEAR(s->parts[LEEKRL_REWARD_TOTAL_LIFE], 0.1));
    memset(parts, 0, sizeof(parts));
    CHECK(!lw_fight_step_finish(&s->fight, 0));
    leekrl_reward_end(&s->tracker, &options, &s->fight, 0, parts);
    CHECK(parts[LEEKRL_REWARD_OUTCOME] == 1);
    CHECK(NEAR(parts[LEEKRL_REWARD_DURATION], 0.2 * 63 / 64));
    CHECK(NEAR(parts[LEEKRL_REWARD_ACTIONS], 0.1 * 100 / 103));
    CHECK(parts[LEEKRL_REWARD_LIFE] == 0);
    memset(parts, 0, sizeof(parts));
    leekrl_reward_end(&s->tracker, &options, &s->fight, 1, parts);
    CHECK(parts[LEEKRL_REWARD_OUTCOME] == -1 && leekrl_reward_sum(parts) == -1);
    stop(s);
    free(s);
    /* Draw at the turn limit. */
    s = duel();
    while (!lw_fight_step_is_over(&s->fight)) {
        LwEntity *actor;
        CHECK(!lw_fight_step_end_turn(&s->fight));
        if (lw_fight_step_is_over(&s->fight)) break;
        do CHECK(!lw_fight_step_begin_turn(&s->fight, &actor));
        while (!actor && !lw_fight_step_is_over(&s->fight));
    }
    CHECK(step(s, 0) == 0);
    CHECK(!lw_fight_step_finish(&s->fight, 0));
    for (i = 0; i < 2; ++i) {
        memset(parts, 0, sizeof(parts));
        leekrl_reward_end(&s->tracker, &options, &s->fight, i, parts);
        CHECK(NEAR(leekrl_reward_sum(parts), -0.2) && NEAR(parts[LEEKRL_REWARD_OUTCOME], -0.2));
    }
    stop(s);
    free(s);
}

/* Random fights: totals of both sides are opposite and equal to the final
 * difference, whenever each side collects its rewards. */
static void test_totals(int fights)
{
    session *s = malloc(sizeof(*s));
    const leekrl_action *actions = leekrl_action_catalog_actions(catalog);
    size_t n = leekrl_action_catalog_count(catalog);
    uint8_t *mask = malloc(n);
    leekrl_build_pool *pool = leekrl_build_pool_create(catalog);
    leekrl_build_options build_options;
    uint64_t rng = 31;
    double largest = 0;
    long outcomes[3] = {0};
    int i;
    CHECK(s && mask && pool);
    leekrl_build_options_init(&build_options);
    for (i = 0; i < fights; ++i) {
        LwEntityInfo info[2];
        double total[2] = {0, 0}, expected = 0;
        LwEntity *actor;
        int team, k;
        current_seed = i + 1;
        for (team = 0; team < 2; ++team) {
            leekrl_build build;
            CHECK(leekrl_build_random(pool, &build_options, &rng, &build));
            leekrl_build_to_entity(&build, team + 1, &info[team]);
        }
        start(s, &info[0], &info[1], current_seed);
        while (!lw_fight_step_is_over(&s->fight)) {
            int plays = 0;
            CHECK(!lw_fight_step_begin_turn(&s->fight, &actor));
            if (!actor) continue;
            while (!lw_entity_is_dead(actor) && !lw_state_is_finished(&s->state) && ++plays < 12) {
                size_t legal = 0, pick, a;
                int result;
                /* This team decides: it collects what happened since. */
                total[actor->team] += step(s, actor->team);
                CHECK(s->parts[LEEKRL_REWARD_OUTCOME] == 0);
                CHECK(leekrl_action_catalog_mask(catalog, &s->fight, actor, mask) ==
                      LEEKRL_ACTION_OK);
                for (a = 0; a < n; ++a) legal += mask[a];
                pick = (size_t)(next_random(&rng) % legal);
                for (a = 0; a < n; ++a)
                    if (mask[a] && pick-- == 0) break;
                if (!a) break;
                CHECK(leekrl_action_execute(&s->fight, actor, &actions[a], &result) ==
                      LEEKRL_ACTION_OK);
            }
            CHECK(!lw_fight_step_end_turn(&s->fight));
        }
        total[0] += step(s, 0);
        total[1] += step(s, 1);
        for (k = 0; k < s->state.n_entities; ++k) {
            const LwEntity *e = s->state.m_entities[k];
            double sign = e->team ? -1 : 1;
            if (e->life <= 0) continue;
            if (lw_entity_is_summon(e)) expected += sign * (0.01 + 0.01 * e->life / e->m_total_life);
            else expected += sign * (0.3 * e->life + 0.1 * e->m_total_life) /
                             info[e->team].life;
        }
        CHECK(fabs(total[0] - expected) < 1e-4 && fabs(total[0] + total[1]) < 1e-4);
        if (fabs(expected) > largest) largest = fabs(expected);
        CHECK(!lw_fight_step_finish(&s->fight, 0));
        ++outcomes[lw_fight_get_winner(&s->fight) + 1];
        stop(s);
    }
    printf("%d fights: %ld draws, %ld and %ld wins, largest score difference %.2f\n", fights,
           outcomes[0], outcomes[1], outcomes[2], largest);
    CHECK(outcomes[0] && outcomes[1] && outcomes[2]);
    leekrl_build_pool_destroy(pool);
    free(mask);
    free(s);
}

/* Through the environment: the reward is the sum of its parts. */
static void test_battle(void)
{
    LwEntityInfo info[2];
    leekrl_battle_options battle_options;
    leekrl_battle *battle = NULL;
    leekrl_env *env = NULL;
    leekrl_step_result result;
    double total = 0, parts_total = 0;
    int i;
    for (i = 0; i < 2; ++i) {
        lw_entity_info_init(&info[i]);
        info[i].id = i + 1;
        info[i].level = 301;
        info[i].life = 1000;
        info[i].tp = 10;
        info[i].mp = 5;
        info[i].ram = 6;
    }
    leekrl_battle_options_init(&battle_options);
    battle_options.max_turns = 3;
    CHECK(leekrl_battle_create(&info[0], &info[1], &battle_options, &battle) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_create(leekrl_battle_observation_size(battle),
                            leekrl_action_catalog_count(leekrl_battle_catalog(battle)),
                            leekrl_battle_callbacks(), battle, &env) == LEEKRL_ENV_OK);
    CHECK(leekrl_env_reset(env, 2) == LEEKRL_ENV_OK);
    do {
        CHECK(leekrl_env_step(env, 0, &result) == LEEKRL_ENV_OK);
        total += result.reward;
        parts_total += leekrl_reward_sum(leekrl_battle_reward_parts(battle));
    } while (!result.terminated);
    CHECK(NEAR(total, -0.2) && NEAR(parts_total, total));
    CHECK(NEAR(leekrl_battle_reward_parts(battle)[LEEKRL_REWARD_OUTCOME], -0.2));
    leekrl_env_destroy(env);
    leekrl_battle_destroy(battle);
    battle = NULL;
    battle_options.rewards.action_scale = 0;
    CHECK(leekrl_battle_create(&info[0], &info[1], &battle_options, &battle) ==
          LEEKRL_ENV_INVALID_ARGUMENT);
    battle_options.rewards.action_scale = 1;
    battle_options.rewards.life = NAN;
    CHECK(leekrl_battle_create(&info[0], &info[1], &battle_options, &battle) ==
          LEEKRL_ENV_INVALID_ARGUMENT);
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
    leekrl_reward_options_init(&options);
    CHECK(leekrl_reward_options_valid(&options));
    test_life();
    test_summons();
    test_end();
    test_totals(argc > 1 ? atoi(argv[1]) : 100);
    test_battle();
    leekrl_action_catalog_destroy(catalog);
    leekrl_item_snapshot_clear();
    return 0;
}
