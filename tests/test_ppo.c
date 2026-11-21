#include "leekrl/ppo.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
#define OK(x) CHECK((x) == LR_OK)

static void close_to(double a, double b, double tolerance) {
    CHECK(fabs(a - b) <= tolerance);
}

static void test_masks_and_reproducibility(void) {
    lr_rng a, b;
    lr_ppo *m1, *m2;
    unsigned char mask[] = {1, 0, 1, 0}, none[] = {0, 0, 0, 0};
    double obs[] = {0.5, -0.75}, p[4], lp, v, ent, lp2, v2;
    size_t i, x, y;
    uint64_t state;
    lr_rng_seed(&a, 123); lr_rng_seed(&b, 123);
    OK(lr_ppo_create(&m1, 2, 5, 4, &a));
    OK(lr_ppo_create(&m2, 2, 5, 4, &b));
    OK(lr_ppo_evaluate(m1, obs, mask, 0, p, &lp, &v, &ent));
    OK(lr_ppo_value(m1, obs, &v2));
    close_to(v, v2, 0);
    close_to(p[0] + p[2], 1, 1e-14);
    CHECK(p[1] == 0 && p[3] == 0 && ent > 0);
    close_to(exp(lp), p[0], 1e-14);
    CHECK(lr_ppo_evaluate(m1, obs, mask, 1, NULL, NULL, NULL, NULL) == LR_INVALID);
    state = a.state;
    CHECK(lr_ppo_sample(m1, obs, none, &a, &x, NULL, NULL) == LR_INVALID);
    CHECK(state == a.state);
    CHECK(lr_ppo_evaluate(m1, obs, none, 0, NULL, NULL, NULL, NULL) == LR_INVALID);
    for (i = 0; i < 1000; ++i) {
        OK(lr_ppo_sample(m1, obs, mask, &a, &x, &lp, &v));
        OK(lr_ppo_sample(m2, obs, mask, &b, &y, &lp2, &v2));
        CHECK((x == 0 || x == 2) && x == y && lp == lp2 && v == v2);
        OK(lr_ppo_evaluate(m1, obs, mask, x, p, &lp2, NULL, NULL));
        close_to(lp, lp2, 0);
    }
    obs[0] = NAN;
    CHECK(lr_ppo_sample(m1, obs, mask, &a, &x, NULL, NULL) == LR_INVALID);
    CHECK(lr_ppo_value(m1, obs, &v2) == LR_INVALID);
    lr_ppo_destroy(m1); lr_ppo_destroy(m2);
}

static void check_target(lr_trajectory *t, size_t i, double advantage, double target) {
    double a, r;
    OK(lr_trajectory_target(t, i, &a, &r));
    close_to(a, advantage, 1e-12); close_to(r, target, 1e-12);
}

static void test_gae(void) {
    lr_trajectory *t;
    double obs[] = {1};
    unsigned char mask[] = {1};
    OK(lr_trajectory_create(&t, 5, 1, 1));
    /* Two episodes plus an unfinished rollout. The third step must not leak
       backward across either boundary. */
    OK(lr_trajectory_push(t, obs, mask, 0, 0, 0.5, 1, 0.7, 0, 0));
    OK(lr_trajectory_push(t, obs, mask, 0, 0, 0.7, 2, 99, 1, 0));
    OK(lr_trajectory_push(t, obs, mask, 0, 0, 1.0, 3, 4, 0, 1));
    OK(lr_trajectory_push(t, obs, mask, 0, 0, 2.0, 5, 6, 0, 0));
    CHECK(lr_trajectory_target(t, 0, NULL, &obs[0]) == LR_INVALID);
    OK(lr_trajectory_gae(t, 0.9, 0.8));
    check_target(t, 1, 1.3, 2.0);
    check_target(t, 0, 1 + .9 * .7 - .5 + .9 * .8 * 1.3,
                 .5 + 1 + .9 * .7 - .5 + .9 * .8 * 1.3);
    check_target(t, 2, 3 + .9 * 4 - 1, 3 + .9 * 4);
    check_target(t, 3, 5 + .9 * 6 - 2, 5 + .9 * 6);
    CHECK(lr_trajectory_push(t, obs, mask, 0, 0, 0, 0, 0, 2, 1) == LR_INVALID);
    CHECK(lr_trajectory_size(t) == 4);
    OK(lr_trajectory_push(t, obs, mask, 0, 0, 1, 2, 99, 1, 1));
    CHECK(lr_trajectory_target(t, 0, &obs[0], NULL) == LR_INVALID);
    OK(lr_trajectory_gae(t, 0.9, 0.8));
    check_target(t, 4, 1, 2);
    lr_trajectory_clear(t);
    CHECK(lr_trajectory_size(t) == 0);
    lr_trajectory_destroy(t);
}

static void test_clipping(void) {
    lr_rng a, b;
    lr_ppo *m1, *m2;
    lr_trajectory *t;
    double obs[] = {1}, lp, value, p1[2], p2[2];
    unsigned char mask[] = {1, 1};
    lr_ppo_options opt = {0.05, 0.0, 0.0, 0.0, 1};
    int sign;
    for (sign = -1; sign <= 1; sign += 2) {
        opt.epochs = 1;
        lr_rng_seed(&a, 9); lr_rng_seed(&b, 9);
        OK(lr_ppo_create(&m1, 1, 4, 2, &a));
        OK(lr_ppo_create(&m2, 1, 4, 2, &b));
        OK(lr_trajectory_create(&t, 1, 1, 2));
        OK(lr_ppo_evaluate(m1, obs, mask, 0, NULL, &lp, &value, NULL));
        OK(lr_trajectory_push(t, obs, mask, 0, lp, value, value + sign, 0, 1, 0));
        OK(lr_trajectory_gae(t, 1, 1));
        OK(lr_ppo_update(m1, t, &opt));
        opt.epochs = 8;
        OK(lr_ppo_update(m2, t, &opt));
        OK(lr_ppo_evaluate(m1, obs, mask, 0, p1, NULL, NULL, NULL));
        OK(lr_ppo_evaluate(m2, obs, mask, 0, p2, NULL, NULL, NULL));
        close_to(p1[0], p2[0], 1e-14);
        CHECK(sign > 0 ? p1[0] > exp(lp) : p1[0] < exp(lp));
        lr_trajectory_destroy(t); lr_ppo_destroy(m1); lr_ppo_destroy(m2);
    }
}

static void test_entropy(void) {
    lr_rng rng;
    lr_ppo *m;
    lr_trajectory *t;
    double obs[] = {1}, lp, value, before, after;
    unsigned char mask[] = {1, 1, 0};
    lr_ppo_options opt = {0.1, 0.2, 0.0, 1.0, 8};
    lr_rng_seed(&rng, 91);
    OK(lr_ppo_create(&m, 1, 4, 3, &rng));
    OK(lr_trajectory_create(&t, 1, 1, 3));
    OK(lr_ppo_evaluate(m, obs, mask, 0, NULL, &lp, &value, &before));
    OK(lr_trajectory_push(t, obs, mask, 0, lp, value, value, 0, 1, 0));
    OK(lr_trajectory_gae(t, 1, 1));
    OK(lr_ppo_update(m, t, &opt));
    OK(lr_ppo_evaluate(m, obs, mask, 0, NULL, NULL, NULL, &after));
    CHECK(after > before);
    lr_trajectory_destroy(t); lr_ppo_destroy(m);
}

static void test_bandit(void) {
    lr_rng rng;
    lr_ppo *m;
    lr_trajectory *t;
    double obs[] = {1}, p[3], lp, v, before, initial_value;
    unsigned char mask[] = {1, 1, 0};
    lr_ppo_options opt = {0.08, 0.2, 0.5, 0.01, 4};
    size_t iteration, i, action;
    lr_rng_seed(&rng, 47);
    OK(lr_ppo_create(&m, 1, 8, 3, &rng));
    OK(lr_trajectory_create(&t, 64, 1, 3));
    OK(lr_ppo_evaluate(m, obs, mask, 1, p, NULL, &initial_value, NULL));
    before = p[1];
    for (iteration = 0; iteration < 100; ++iteration) {
        lr_trajectory_clear(t);
        for (i = 0; i < 64; ++i) {
            OK(lr_ppo_sample(m, obs, mask, &rng, &action, &lp, &v));
            CHECK(action < 2);
            OK(lr_trajectory_push(t, obs, mask, action, lp, v,
                                  action == 1 ? 1.0 : -1.0, 0, 1, 0));
        }
        OK(lr_trajectory_gae(t, 0.99, 0.95));
        OK(lr_ppo_update(m, t, &opt));
    }
    OK(lr_ppo_evaluate(m, obs, mask, 1, p, NULL, &v, NULL));
    CHECK(p[1] > 0.95 && p[1] > before + 0.1 && p[2] == 0);
    CHECK(fabs(v - 1.0) < fabs(initial_value - 1.0));
    lr_trajectory_destroy(t); lr_ppo_destroy(m);
}

int main(void) {
    test_masks_and_reproducibility();
    test_gae();
    test_clipping();
    test_entropy();
    test_bandit();
    puts("PPO tests passed");
    return 0;
}
