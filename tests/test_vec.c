#include "leekrl/item_snapshot.h"
#include "leekrl/rules.h"
#include "leekrl/symmetry.h"
#include "leekrl/threads.h"
#include "leekrl/vec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

typedef struct {
    leekrl_vec_options options;
    leekrl_vec_buffers b;
    leekrl_vec *vec;
    int32_t *actions;
    size_t n, floats, count;
} session;

static uint64_t next_random(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static void open_session(session *s, int envs, int threads, uint64_t seed)
{
    char error[256];
    memset(&s->b, 0, sizeof(s->b));
    CHECK(leekrl_vec_sizes(&s->count, &s->floats));
    CHECK(s->floats == 2705 && s->count == 10566);
    s->n = (size_t)envs;
    s->options.envs = envs;
    s->options.threads = threads;
    s->options.seed = seed;
    s->b.floats = malloc(s->n * s->floats * sizeof(float));
    s->b.map = malloc(s->n * LEEKRL_OBS_MAP_CELLS);
    s->b.legal = malloc(s->n * s->count * sizeof(int16_t));
    s->b.legal_counts = calloc(s->n, sizeof(int32_t));
    s->b.held = malloc(s->n * LEEKRL_OBS_MAX_HELD * sizeof(int16_t));
    s->b.held_counts = calloc(s->n, sizeof(int32_t));
    s->b.place = calloc(s->n, sizeof(int16_t));
    s->b.symmetry = calloc(s->n, 1);
    s->b.rewards = calloc(s->n * LEEKRL_REWARD_PARTS, sizeof(float));
    s->b.team = calloc(s->n, 1);
    s->b.first = calloc(s->n, 1);
    s->b.done = calloc(s->n, 1);
    s->b.end_rewards = calloc(s->n * 2 * LEEKRL_REWARD_PARTS, sizeof(float));
    s->b.episode = calloc(s->n * LEEKRL_EPISODE_SIZE, sizeof(float));
    s->actions = calloc(s->n, sizeof(*s->actions));
    CHECK(s->b.floats && s->b.map && s->b.legal && s->b.held && s->b.episode && s->actions);
    s->vec = leekrl_vec_create(&s->options, &s->b, error, sizeof(error));
    if (!s->vec) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }
    CHECK(leekrl_vec_floats(s->vec) == s->floats && leekrl_vec_actions(s->vec) == s->count);
}

static void close_session(session *s)
{
    leekrl_vec_destroy(s->vec);
    free(s->b.floats);
    free(s->b.map);
    free(s->b.legal);
    free(s->b.legal_counts);
    free(s->b.held);
    free(s->b.held_counts);
    free(s->b.place);
    free(s->b.symmetry);
    free(s->b.rewards);
    free(s->b.team);
    free(s->b.first);
    free(s->b.done);
    free(s->b.end_rewards);
    free(s->b.episode);
    free(s->actions);
}

/* Legal actions of an environment. */
static const int16_t *legal_of(const session *s, size_t env)
{
    const int16_t *list = s->b.legal;
    size_t i;
    for (i = 0; i < env; ++i) list += s->b.legal_counts[i];
    return list;
}

/* A random legal action, decided by the seed and what the environment shows. */
static int32_t choose(const session *s, size_t env, uint64_t *rng)
{
    const int16_t *list = legal_of(s, env);
    int32_t n = s->b.legal_counts[env], i;
    CHECK(n > 0 && list[0] == 0);
    for (i = 1; i < n; ++i) CHECK(list[i] > list[i - 1] && (size_t)list[i] < s->count);
    return list[next_random(rng) % (uint64_t)n];
}

static void must(session *s, int ok)
{
    if (ok) return;
    fprintf(stderr, "vec: %s\n", leekrl_vec_error(s->vec));
    exit(1);
}

static void hash_bytes(uint64_t *hash, const void *data, size_t size)
{
    const unsigned char *bytes = data;
    while (size--) *hash = (*hash ^ *bytes++) * UINT64_C(0x100000001B3);
}

static uint64_t play_hash(int envs, int threads, uint64_t seed, int steps, double known)
{
    session s;
    uint64_t hash = UINT64_C(0xCBF29CE484222325), rng = 9;
    size_t i, legal, held;
    leekrl_vec_options_init(&s.options);
    s.options.known_probability = known;
    open_session(&s, envs, threads, seed);
    must(&s, leekrl_vec_reset(s.vec));
    while (steps--) {
        hash_bytes(&hash, s.b.floats, s.n * s.floats * sizeof(float));
        hash_bytes(&hash, s.b.map, s.n * LEEKRL_OBS_MAP_CELLS);
        hash_bytes(&hash, s.b.legal_counts, s.n * sizeof(int32_t));
        hash_bytes(&hash, s.b.held_counts, s.n * sizeof(int32_t));
        for (i = legal = held = 0; i < s.n; ++i) {
            legal += (size_t)s.b.legal_counts[i];
            held += (size_t)s.b.held_counts[i];
        }
        hash_bytes(&hash, s.b.legal, legal * sizeof(int16_t));
        hash_bytes(&hash, s.b.held, held * sizeof(int16_t));
        hash_bytes(&hash, s.b.place, s.n * sizeof(int16_t));
        hash_bytes(&hash, s.b.symmetry, s.n);
        hash_bytes(&hash, s.b.rewards, s.n * LEEKRL_REWARD_PARTS * sizeof(float));
        hash_bytes(&hash, s.b.team, s.n);
        hash_bytes(&hash, s.b.first, s.n);
        hash_bytes(&hash, s.b.done, s.n);
        hash_bytes(&hash, s.b.end_rewards, s.n * 2 * LEEKRL_REWARD_PARTS * sizeof(float));
        hash_bytes(&hash, s.b.episode, s.n * LEEKRL_EPISODE_SIZE * sizeof(float));
        for (i = 0; i < s.n; ++i) s.actions[i] = choose(&s, i, &rng);
        must(&s, leekrl_vec_step(s.vec, s.actions));
    }
    close_session(&s);
    return hash;
}

/* Same seed: same bytes, whatever the threads. Other seed: other bytes. */
static void test_determinism(void)
{
    uint64_t one = play_hash(24, 1, 5, 400, 0.01);
    CHECK(play_hash(24, 1, 5, 400, 0.01) == one);
    CHECK(play_hash(24, 3, 5, 400, 0.01) == one);
    CHECK(play_hash(24, 16, 5, 400, 0.01) == one);
    CHECK(play_hash(24, 1, 6, 400, 0.01) != one);
}

/* Rewards collected decision by decision give the final difference of scores,
 * outcomes are consistent, and both teams play. */
static void test_fights(void)
{
    session s;
    uint64_t rng = 3;
    const int envs = 64;
    double (*total)[2] = calloc((size_t)envs, sizeof(*total));
    int (*decisions)[2] = calloc((size_t)envs, sizeof(*decisions));
    long fights = 0, wins[3] = {0}, steps = 0, first = 0, summon_decisions = 0;
    double turns = 0;
    int step;
    size_t i;
    CHECK(total && decisions);
    leekrl_vec_options_init(&s.options);
    open_session(&s, envs, 4, 12);
    CHECK(!leekrl_vec_step(s.vec, s.actions)); /* not reset */
    must(&s, leekrl_vec_reset(s.vec));
    for (i = 0; i < s.n; ++i) CHECK(!s.b.done[i] && s.b.first[i]);
    for (step = 0; step < 3000; ++step) {
        for (i = 0; i < s.n; ++i) {
            const float *e = s.b.episode + i * LEEKRL_EPISODE_SIZE;
            const float *parts = s.b.rewards + i * LEEKRL_REWARD_PARTS;
            const float *f = s.b.floats + i * s.floats;
            int team = s.b.team[i], k, t;
            if (s.b.done[i]) {
                int winner = (int)e[LEEKRL_EPISODE_WINNER];
                ++fights;
                ++wins[winner + 1];
                turns += e[LEEKRL_EPISODE_TURNS];
                CHECK(winner >= -1 && winner <= 1);
                CHECK(e[LEEKRL_EPISODE_TURNS] >= 1 && e[LEEKRL_EPISODE_TURNS] <= 65);
                CHECK((winner == -1) == (e[LEEKRL_EPISODE_TURNS] == 65 ||
                      (e[LEEKRL_EPISODE_LIFE] == 0 && e[LEEKRL_EPISODE_LIFE + 1] == 0)));
                for (t = 0; t < 2; ++t) {
                    const float *end = s.b.end_rewards + (2 * i + (size_t)t) * LEEKRL_REWARD_PARTS;
                    float outcome = winner == -1 ? -0.2f : winner == t ? 1.0f : -1.0f;
                    CHECK(end[LEEKRL_REWARD_OUTCOME] == outcome);
                    /* Nothing for a win during the last turn. */
                    CHECK((end[LEEKRL_REWARD_DURATION] > 0) ==
                          (winner == t && e[LEEKRL_EPISODE_TURNS] < 64));
                    CHECK((end[LEEKRL_REWARD_ACTIONS] > 0) == (winner == t));
                    CHECK(e[LEEKRL_EPISODE_DECISIONS + t] == decisions[i][t]);
                    CHECK(e[LEEKRL_EPISODE_ACTIONS + t] <= decisions[i][t]);
                    CHECK((e[LEEKRL_EPISODE_LIFE + t] > 0) == (winner == t ||
                          (winner == -1 && e[LEEKRL_EPISODE_LIFE + t] > 0)));
                    if (decisions[i][t])
                        for (k = 0; k < LEEKRL_REWARD_OUTCOME; ++k) total[i][t] += end[k];
                }
                /* Opposite, up to what preceded each side's first decision. */
                CHECK(fabs(total[i][0]) < 3 && fabs(total[i][1]) < 3);
                total[i][0] = total[i][1] = 0;
                decisions[i][0] = decisions[i][1] = 0;
            }
            CHECK(s.b.first[i] == (decisions[i][team] == 0));
            for (k = 0; k < LEEKRL_REWARD_PARTS; ++k) {
                CHECK(isfinite(parts[k]));
                if (s.b.first[i] || k >= LEEKRL_REWARD_OUTCOME) CHECK(parts[k] == 0);
                else total[i][team] += parts[k];
            }
            ++decisions[i][team];
            first += s.b.first[i];
            /* The acting entity is present, an ally, at the center. */
            CHECK(f[LEEKRL_OBS_ENTITIES] == 1 && f[LEEKRL_OBS_ENTITIES + LEEKRL_OBS_ALLY] == 1);
            CHECK(s.b.map[i * LEEKRL_OBS_MAP_CELLS + LEEKRL_OBS_MAP_CENTER] == LEEKRL_CELL_SELF);
            summon_decisions += f[LEEKRL_OBS_ENTITIES + LEEKRL_OBS_TYPE] == 0;
            s.actions[i] = choose(&s, i, &rng);
            ++steps;
        }
        must(&s, leekrl_vec_step(s.vec, s.actions));
    }
    printf("%ld decisions (%ld by summons), %ld fights, %.1f turns, %ld draws, %ld / %ld wins\n",
           steps, summon_decisions, fights, turns / (double)fights, wins[0], wins[1], wins[2]);
    CHECK(fights > 50 && wins[0] && wins[1] && wins[2] && summon_decisions);
    CHECK(first >= 2 * fights - 5);
    close_session(&s);
    free(total);
    free(decisions);
}

/* With forced turns kept, some decisions have a single choice. */
static void test_forced(void)
{
    session s;
    uint64_t rng = 8;
    long forced[2] = {0, 0};
    int skip, step;
    for (skip = 0; skip < 2; ++skip) {
        leekrl_vec_options_init(&s.options);
        s.options.skip_forced = skip;
        open_session(&s, 8, 2, 4);
        must(&s, leekrl_vec_reset(s.vec));
        for (step = 0; step < 1500; ++step) {
            size_t i;
            for (i = 0; i < s.n; ++i) {
                forced[skip] += s.b.legal_counts[i] == 1;
                s.actions[i] = choose(&s, i, &rng);
            }
            must(&s, leekrl_vec_step(s.vec, s.actions));
        }
        close_session(&s);
    }
    printf("decisions with a single choice: %ld kept, %ld once skipped\n", forced[0], forced[1]);
    CHECK(forced[0] > 0 && forced[1] == 0);
}

/* Turned, the same actions give the same fights, seen turned. */
static void test_facing(void)
{
    session a, b;
    leekrl_action_catalog *catalog;
    const leekrl_action *actions;
    uint64_t rng = 21;
    long met[LEEKRL_SYMMETRIES] = {0};
    const int size = LEEKRL_OBS_ENTITY_SIZE;
    int step;
    CHECK(leekrl_action_catalog_create_without(leekrl_solo_excluded_items,
              leekrl_solo_excluded_count, &catalog) == LEEKRL_ACTION_OK);
    actions = leekrl_action_catalog_actions(catalog);
    leekrl_vec_options_init(&a.options);
    leekrl_vec_options_init(&b.options);
    CHECK(a.options.face_enemy);
    a.options.face_enemy = 0;
    open_session(&a, 24, 4, 33);
    open_session(&b, 24, 4, 33);
    must(&a, leekrl_vec_reset(a.vec));
    must(&b, leekrl_vec_reset(b.vec));
    for (step = 0; step < 1500; ++step) {
        size_t i, k;
        for (i = 0; i < a.n; ++i) {
            const float *f = a.b.floats + i * a.floats, *g = b.b.floats + i * b.floats;
            const int16_t *list = legal_of(&a, i), *turned = legal_of(&b, i);
            int symmetry = b.b.symmetry[i], n = a.b.legal_counts[i], enemy = 0, e, j;
            uint8_t seen[10566] = {0};
            leekrl_action action;
            ++met[symmetry];
            CHECK(a.b.symmetry[i] == 0 && symmetry < LEEKRL_SYMMETRIES);
            CHECK(n == b.b.legal_counts[i] && a.b.held_counts[i] == b.b.held_counts[i]);
            for (k = 0; k < (size_t)n; ++k) seen[turned[k]] = 1;
            for (k = 0; k < (size_t)n; ++k) {
                action = actions[list[k]];
                leekrl_turn(symmetry, &action.dx, &action.dy);
                CHECK(seen[leekrl_action_catalog_find(catalog, &action)]);
            }
            for (k = 0; k < a.floats; ++k) {
                int offset = (int)k - LEEKRL_OBS_ENTITIES, field = offset % size;
                int dx, dy;
                if (offset < 0 || k >= LEEKRL_OBS_ITEMS ||
                    (field != LEEKRL_OBS_DX && field != LEEKRL_OBS_DY)) {
                    CHECK(f[k] == g[k]);
                } else if (field == LEEKRL_OBS_DX) {
                    dx = (int)lround(f[k] * LEEKRL_OBS_MAP_RADIUS);
                    dy = (int)lround(f[k + 1] * LEEKRL_OBS_MAP_RADIUS);
                    leekrl_turn(symmetry, &dx, &dy);
                    CHECK((int)lround(g[k] * LEEKRL_OBS_MAP_RADIUS) == dx);
                    CHECK((int)lround(g[k + 1] * LEEKRL_OBS_MAP_RADIUS) == dy);
                }
            }
            for (e = 0; e < LEEKRL_OBS_MAX_ENTITIES; ++e) {
                const float *block = g + LEEKRL_OBS_ENTITIES + e * size;
                if (!block[LEEKRL_OBS_PRESENT] || block[LEEKRL_OBS_ALLY] ||
                    !block[LEEKRL_OBS_TYPE]) continue;
                ++enemy;
                CHECK(block[LEEKRL_OBS_DY] >= 0 &&
                      block[LEEKRL_OBS_DX] >= block[LEEKRL_OBS_DY]);
            }
            CHECK(enemy == 1);
            j = (int)(next_random(&rng) % (uint64_t)n);
            a.actions[i] = list[j];
            action = actions[list[j]];
            leekrl_turn(symmetry, &action.dx, &action.dy);
            b.actions[i] = (int32_t)leekrl_action_catalog_find(catalog, &action);
        }
        must(&a, leekrl_vec_step(a.vec, a.actions));
        must(&b, leekrl_vec_step(b.vec, b.actions));
        CHECK(!memcmp(a.b.team, b.b.team, a.n) && !memcmp(a.b.first, b.b.first, a.n) &&
              !memcmp(a.b.done, b.b.done, a.n));
        CHECK(!memcmp(a.b.rewards, b.b.rewards, a.n * LEEKRL_REWARD_PARTS * sizeof(float)));
        CHECK(!memcmp(a.b.end_rewards, b.b.end_rewards,
                      a.n * 2 * LEEKRL_REWARD_PARTS * sizeof(float)));
        CHECK(!memcmp(a.b.episode, b.b.episode, a.n * LEEKRL_EPISODE_SIZE * sizeof(float)));
    }
    for (step = 0; step < LEEKRL_SYMMETRIES; ++step) CHECK(met[step] > 100);
    close_session(&a);
    close_session(&b);
    leekrl_action_catalog_destroy(catalog);
}

/* An entity that has just moved cannot move, but if a plant can awaken. */
static void test_repeated_moves(void)
{
    session s;
    uint64_t rng = 5;
    long again[2] = {0, 0}, plants = 0, decisions[2] = {0, 0};
    const int size = LEEKRL_OBS_ENTITY_SIZE;
    int repeat, step;
    for (repeat = 0; repeat < 2; ++repeat) {
        /* Who decided last, if it moved: its block, team and turn. */
        float (*last)[LEEKRL_OBS_ENTITY_SIZE] = malloc(64 * sizeof(*last));
        int moved[64] = {0}, team[64] = {0};
        float turn[64] = {0};
        CHECK(last);
        leekrl_vec_options_init(&s.options);
        CHECK(!s.options.repeat_moves);
        s.options.repeat_moves = repeat;
        open_session(&s, 64, 4, 77);
        must(&s, leekrl_vec_reset(s.vec));
        for (step = 0; step < 4000; ++step) {
            size_t i;
            for (i = 0; i < s.n; ++i) {
                const float *f = s.b.floats + i * s.floats, *me = f + LEEKRL_OBS_ENTITIES;
                const int16_t *list = legal_of(&s, i);
                int moves = s.b.legal_counts[i] > 1 && list[1] <= LEEKRL_MOVE_ACTION_COUNT, e;
                int plant = 0, same;
                /* Near enough to have been in reach of the first move. */
                for (e = 0; e < LEEKRL_OBS_MAX_ENTITIES; ++e)
                    plant |= me[e * size + LEEKRL_OBS_ZONE] > 0 &&
                        me[e * size + LEEKRL_OBS_DISTANCE] * LEEKRL_OBS_MAP_RADIUS <=
                        me[e * size + LEEKRL_OBS_ZONE] * 10 + 2 * last[i][LEEKRL_OBS_MP] * 15 + 0.5f;
                same = moved[i] && !s.b.done[i] && team[i] == s.b.team[i] && turn[i] == f[LEEKRL_OBS_TURN] &&
                       !memcmp(last[i] + LEEKRL_OBS_TYPE, me + LEEKRL_OBS_TYPE,
                               (1 + LEEKRL_OBS_SUMMON_TYPES) * sizeof(float)) &&
                       last[i][LEEKRL_OBS_TOTAL_LIFE] == me[LEEKRL_OBS_TOTAL_LIFE] &&
                       me[LEEKRL_OBS_MP] < last[i][LEEKRL_OBS_MP] &&
                       me[LEEKRL_OBS_TP] == last[i][LEEKRL_OBS_TP] &&
                       me[LEEKRL_OBS_MP] < me[LEEKRL_OBS_TOTAL_MP] &&
                       last[i][LEEKRL_OBS_MP] * 15 < LEEKRL_MAX_MOVE_DISTANCE + 0.5f;
                ++decisions[repeat];
                if (same && moves) {
                    ++again[repeat];
                    plants += !repeat;
                    if (!repeat) CHECK(plant);
                }
                s.actions[i] = choose(&s, i, &rng);
                moved[i] = s.actions[i] >= 1 && s.actions[i] <= LEEKRL_MOVE_ACTION_COUNT;
                memcpy(last[i], me, sizeof(last[i]));
                team[i] = s.b.team[i];
                turn[i] = f[LEEKRL_OBS_TURN];
            }
            must(&s, leekrl_vec_step(s.vec, s.actions));
        }
        close_session(&s);
        free(last);
    }
    printf("moves possible right after a move: %.1f%% of decisions allowed, %.2f%% not (plants)\n",
           100.0 * (double)again[1] / (double)decisions[1],
           100.0 * (double)again[0] / (double)decisions[0]);
    CHECK(again[1] > decisions[1] / 5 && again[0] < again[1] / 3 && plants);
}

static void test_known_builds(void)
{
    session s;
    leekrl_action_catalog *catalog;
    leekrl_build_pool *pool;
    leekrl_build_options build_options;
    leekrl_build builds[2], bad;
    uint64_t rng = 1;
    char error[128];
    long known[3] = {0}, fights = 0, drawn;
    int step;
    size_t i;
    double probabilities[] = {1, 0.3};
    size_t p;
    CHECK(leekrl_action_catalog_create(&catalog) == LEEKRL_ACTION_OK);
    pool = leekrl_build_pool_create(catalog);
    CHECK(pool);
    leekrl_build_options_init(&build_options);
    /* From the unrestricted pool, so that some are refused below. */
    do CHECK(leekrl_build_random(pool, &build_options, &rng, &bad));
    while (bad.weapons[0] != 410);
    leekrl_build_pool_destroy(pool);
    leekrl_action_catalog_destroy(catalog);
    CHECK(leekrl_action_catalog_create_without((const int[]){84, 408, 409, 410, 411, 412, 413, 414,
          415, 416, 417, 418, 419, 425, 448}, 15, &catalog) == LEEKRL_ACTION_OK);
    pool = leekrl_build_pool_create(catalog);
    CHECK(leekrl_build_random(pool, &build_options, &rng, &builds[0]));
    CHECK(leekrl_build_random(pool, &build_options, &rng, &builds[1]));
    leekrl_build_pool_destroy(pool);
    leekrl_action_catalog_destroy(catalog);
    for (p = 0; p < 2; ++p) {
        leekrl_vec_options_init(&s.options);
        s.options.known_probability = probabilities[p];
        open_session(&s, 32, 4, 2);
        CHECK(!leekrl_vec_add_build(s.vec, &bad, 1, error, sizeof(error)) && error[0]);
        CHECK(!leekrl_vec_add_build(s.vec, &builds[0], 0, error, sizeof(error)));
        CHECK(!leekrl_vec_add_build(s.vec, &builds[0], NAN, error, sizeof(error)));
        CHECK(leekrl_vec_add_build(s.vec, &builds[0], 3, error, sizeof(error)));
        CHECK(leekrl_vec_add_build(s.vec, &builds[1], 1, error, sizeof(error)));
        must(&s, leekrl_vec_reset(s.vec));
        memset(known, 0, sizeof(known));
        fights = 0;
        for (step = 0; step < 4000; ++step) {
            for (i = 0; i < s.n; ++i) {
                if (s.b.done[i]) {
                    const float *e = s.b.episode + i * LEEKRL_EPISODE_SIZE;
                    ++fights;
                    ++known[(int)e[LEEKRL_EPISODE_KNOWN]];
                    ++known[(int)e[LEEKRL_EPISODE_KNOWN + 1]];
                }
                /* Mostly end turns, for many fights. */
                s.actions[i] = next_random(&rng) % 4 ? 0 : choose(&s, i, &rng);
            }
            must(&s, leekrl_vec_step(s.vec, s.actions));
        }
        drawn = known[1] + known[2];
        printf("known builds at %.1f: %ld of %ld sides, %ld / %ld\n", probabilities[p], drawn,
               2 * fights, known[1], known[2]);
        CHECK(fights > 100);
        if (probabilities[p] == 1) CHECK(drawn == 2 * fights);
        else CHECK(fabs((double)drawn / (double)(2 * fights) - 0.3) < 0.06);
        CHECK(fabs((double)known[1] / (double)drawn - 0.75) < 0.1);
        close_session(&s);
    }
}

static void test_errors(void)
{
    session s;
    char error[256];
    leekrl_vec_options bad;
    const int16_t *list;
    int32_t illegal = -1;
    size_t i;
    leekrl_vec_options_init(&s.options);
    open_session(&s, 4, 2, 1);
    bad = s.options;
    bad.envs = 0;
    CHECK(!leekrl_vec_create(&bad, &s.b, error, sizeof(error)) && error[0]);
    bad = s.options;
    bad.known_probability = 1.5;
    CHECK(!leekrl_vec_create(&bad, &s.b, error, sizeof(error)));
    bad = s.options;
    bad.rewards.action_scale = 0;
    CHECK(!leekrl_vec_create(&bad, &s.b, error, sizeof(error)));
    bad = s.options;
    bad.trim = (leekrl_trim)3;
    CHECK(!leekrl_vec_create(&bad, &s.b, error, sizeof(error)));
    must(&s, leekrl_vec_reset(s.vec));
    /* The first action that the list skips. */
    for (i = 0, list = legal_of(&s, 2); illegal < 0; ++i)
        if (i == (size_t)s.b.legal_counts[2] || list[i] != (int16_t)i) illegal = (int32_t)i;
    s.actions[2] = illegal;
    CHECK(!leekrl_vec_step(s.vec, s.actions));
    CHECK(strstr(leekrl_vec_error(s.vec), "environment 2") &&
          strstr(leekrl_vec_error(s.vec), "illegal action"));
    s.actions[2] = 0;
    CHECK(!leekrl_vec_step(s.vec, s.actions)); /* until reset */
    must(&s, leekrl_vec_reset(s.vec));
    must(&s, leekrl_vec_step(s.vec, s.actions));
    s.actions[1] = 1 << 30;
    CHECK(!leekrl_vec_step(s.vec, s.actions));
    close_session(&s);
}

static void count_task(void *context, int index) { ++((int *)context)[index]; }

static void test_threads(void)
{
    int counts[1000] = {0}, i, run;
    leekrl_threads *threads = leekrl_threads_create(7);
    CHECK(threads && !leekrl_threads_create(0));
    for (run = 0; run < 300; ++run) leekrl_threads_run(threads, count_task, counts, 1000);
    leekrl_threads_run(threads, count_task, counts, 3);
    leekrl_threads_run(threads, count_task, counts, 0);
    for (i = 0; i < 1000; ++i) CHECK(counts[i] == 300 + (i < 3));
    leekrl_threads_destroy(threads);
}

int main(void)
{
    char error[512];
    size_t a, b;
    CHECK(!leekrl_vec_sizes(&a, &b));
    if (!leekrl_item_snapshot_register(error, sizeof(error))) {
        fprintf(stderr, "item registration: %s\n", error);
        return 1;
    }
    test_threads();
    test_errors();
    test_determinism();
    test_facing();
    test_repeated_moves();
    test_forced();
    test_known_builds();
    test_fights();
    leekrl_item_snapshot_clear();
    return 0;
}
