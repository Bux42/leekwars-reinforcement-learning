#include "leekrl/item_snapshot.h"
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
    size_t n, floats, mask_bytes, count;
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
    CHECK(leekrl_vec_sizes(&s->count, &s->floats, &s->mask_bytes));
    CHECK(s->floats == 2705 && s->count == 10566 && s->mask_bytes == 1321);
    s->n = (size_t)envs;
    s->options.envs = envs;
    s->options.threads = threads;
    s->options.seed = seed;
    s->b.floats = malloc(s->n * s->floats * sizeof(float));
    s->b.map = malloc(s->n * LEEKRL_OBS_MAP_CELLS);
    s->b.mask = malloc(s->n * s->mask_bytes);
    s->b.rewards = calloc(s->n * LEEKRL_REWARD_PARTS, sizeof(float));
    s->b.team = calloc(s->n, 1);
    s->b.first = calloc(s->n, 1);
    s->b.done = calloc(s->n, 1);
    s->b.end_rewards = calloc(s->n * 2 * LEEKRL_REWARD_PARTS, sizeof(float));
    s->b.episode = calloc(s->n * LEEKRL_EPISODE_SIZE, sizeof(float));
    s->actions = calloc(s->n, sizeof(*s->actions));
    CHECK(s->b.floats && s->b.map && s->b.mask && s->b.episode && s->actions);
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
    free(s->b.mask);
    free(s->b.rewards);
    free(s->b.team);
    free(s->b.first);
    free(s->b.done);
    free(s->b.end_rewards);
    free(s->b.episode);
    free(s->actions);
}

/* A random legal action, decided by the seed and what the environment shows. */
static int32_t choose(const session *s, size_t env, uint64_t *rng)
{
    const uint8_t *mask = s->b.mask + env * s->mask_bytes;
    size_t i, legal = 0, pick;
    for (i = 0; i < s->count; ++i) legal += (mask[i >> 3] >> (i & 7)) & 1;
    CHECK(legal > 0 && (mask[0] & 1));
    pick = (size_t)(next_random(rng) % legal);
    for (i = 0; i < s->count; ++i)
        if (((mask[i >> 3] >> (i & 7)) & 1) && pick-- == 0) return (int32_t)i;
    return -1;
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
    size_t i;
    leekrl_vec_options_init(&s.options);
    s.options.known_probability = known;
    open_session(&s, envs, threads, seed);
    must(&s, leekrl_vec_reset(s.vec));
    while (steps--) {
        hash_bytes(&hash, s.b.floats, s.n * s.floats * sizeof(float));
        hash_bytes(&hash, s.b.map, s.n * LEEKRL_OBS_MAP_CELLS);
        hash_bytes(&hash, s.b.mask, s.n * s.mask_bytes);
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
                    CHECK((end[LEEKRL_REWARD_DURATION] > 0) == (winner == t));
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
            size_t i, k;
            for (i = 0; i < s.n; ++i) {
                const uint8_t *mask = s.b.mask + i * s.mask_bytes;
                int any = mask[0] & ~1;
                for (k = 1; k < s.mask_bytes; ++k) any |= mask[k];
                forced[skip] += !any;
                s.actions[i] = choose(&s, i, &rng);
            }
            must(&s, leekrl_vec_step(s.vec, s.actions));
        }
        close_session(&s);
    }
    printf("decisions with a single choice: %ld kept, %ld once skipped\n", forced[0], forced[1]);
    CHECK(forced[0] > 0 && forced[1] == 0);
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
    must(&s, leekrl_vec_reset(s.vec));
    for (i = 0; i < s.count && illegal < 0; ++i)
        if (!((s.b.mask[2 * s.mask_bytes + (i >> 3)] >> (i & 7)) & 1)) illegal = (int32_t)i;
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
    size_t a, b, c;
    CHECK(!leekrl_vec_sizes(&a, &b, &c));
    if (!leekrl_item_snapshot_register(error, sizeof(error))) {
        fprintf(stderr, "item registration: %s\n", error);
        return 1;
    }
    test_threads();
    test_errors();
    test_determinism();
    test_forced();
    test_known_builds();
    test_fights();
    leekrl_item_snapshot_clear();
    return 0;
}
