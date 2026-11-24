#ifndef LEEKRL_VEC_H
#define LEEKRL_VEC_H

#include "leekrl/build.h"
#include "leekrl/observation.h"
#include "leekrl/reward.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Many 1v1 fights played side by side, the agent controlling both teams. Needs
 * the item snapshot, registered before and unchanged meanwhile.
 *
 * At any time every environment waits for one decision of one entity. A step
 * executes one action per environment, then plays until the next decision. A
 * finished fight is replaced at once by a new one, with new builds, map and
 * starting positions.
 *
 * Both teams' decisions come interleaved. The learner keeps one trajectory per
 * environment and team: the reward of a step belongs to the previous decision
 * of the team that decides now, whichever of its entities took it.
 */
typedef struct {
    int envs;
    int threads;              /* the calling thread included */
    uint64_t seed;
    double known_probability; /* chance of a known build, for each side alone */
    int skip_forced;          /* end turns that have no other legal action */
    int trim_useless;         /* mask casts that cannot affect anything */
    leekrl_build_options builds;
    leekrl_reward_options rewards;
} leekrl_vec_options;

/* 1 env, 1 thread, seed 0, 1% of known builds, forced turns skipped, useless casts
 * masked. */
void leekrl_vec_options_init(leekrl_vec_options *options);

/* Figures of a finished fight. */
enum {
    LEEKRL_EPISODE_TURNS,
    LEEKRL_EPISODE_WINNER,      /* -1 for a draw, else the team */
    LEEKRL_EPISODE_DECISIONS,   /* 2 teams */
    LEEKRL_EPISODE_ACTIONS = LEEKRL_EPISODE_DECISIONS + 2, /* 2 teams, ends of turn excluded */
    LEEKRL_EPISODE_LIFE = LEEKRL_EPISODE_ACTIONS + 2,      /* 2 teams: final / starting life */
    LEEKRL_EPISODE_KNOWN = LEEKRL_EPISODE_LIFE + 2,        /* 2 teams: known build + 1, or 0 */
    LEEKRL_EPISODE_REDRAWN = LEEKRL_EPISODE_KNOWN + 2,     /* maps rejected before the fight */
    LEEKRL_EPISODE_SIZE
};

/*
 * Caller-owned outputs, valid until the next step. N environments.
 *
 * The mask has one bit per action, least significant bit first. rewards holds
 * the parts of the reward of the deciding team's previous decision, zeros if
 * first is set. When done is set, a fight ended during this step: end_rewards
 * holds, for both teams, the parts of the reward of their last decision (to
 * ignore for a team without any decision), and episode describes that fight. The other outputs are those of the new fight.
 */
typedef struct {
    float *floats;      /* N x leekrl_vec_floats */
    uint8_t *map;       /* N x LEEKRL_OBS_MAP_CELLS */
    uint8_t *mask;      /* N x leekrl_vec_mask_bytes */
    float *rewards;     /* N x LEEKRL_REWARD_PARTS */
    uint8_t *team;      /* N: team that decides */
    uint8_t *first;     /* N: first decision of that team in this fight */
    uint8_t *done;      /* N */
    float *end_rewards; /* N x 2 x LEEKRL_REWARD_PARTS */
    float *episode;     /* N x LEEKRL_EPISODE_SIZE */
} leekrl_vec_buffers;

typedef struct leekrl_vec leekrl_vec;

/* NULL on failure, described in error. Fights start with leekrl_vec_reset. */
leekrl_vec *leekrl_vec_create(const leekrl_vec_options *options,
                              const leekrl_vec_buffers *buffers, char *error,
                              size_t error_size);
void leekrl_vec_destroy(leekrl_vec *vec);

size_t leekrl_vec_actions(const leekrl_vec *vec);
size_t leekrl_vec_floats(const leekrl_vec *vec);
size_t leekrl_vec_mask_bytes(const leekrl_vec *vec);

/* Sizes without an environment. 0 if the item snapshot is not registered. */
int leekrl_vec_sizes(size_t *actions, size_t *floats, size_t *mask_bytes);

/* Describe every action: kind, position of its item among the weapons or chips of
 * the observation (-1 for none), dx, dy. Holds 4 x actions values. Returns 1 or 0. */
int leekrl_vec_action_table(int32_t *table);

/* Item ids of the weapons and chips of the observation, in their order. The arrays
 * hold as many values as leekrl_vec_action_table has items. Returns 1 or 0. */
int leekrl_vec_items(int32_t *weapons, int32_t *chips);

/* Sizes a binding can compare with its own. */
typedef struct {
    int map_cells;
    int reward_parts;
    int episode_size;
    int stats;
    int options_size;  /* sizeof leekrl_vec_options */
    int buffers_size;  /* sizeof leekrl_vec_buffers */
    int build_size;    /* sizeof leekrl_build */
} leekrl_vec_layout_info;

void leekrl_vec_layout(leekrl_vec_layout_info *layout);

/* Add a known build, checked with leekrl_build_validate. Before reset. */
int leekrl_vec_add_build(leekrl_vec *vec, const leekrl_build *build, double weight,
                         char *error, size_t error_size);

/* Start every fight again from the seed. done is not set. Returns 1, or 0 on
 * error: see leekrl_vec_error. */
int leekrl_vec_reset(leekrl_vec *vec);

/* actions holds one legal action per environment. Returns 1, or 0 if any
 * environment failed: the vec must then be reset or destroyed. Results depend
 * on the seed and the actions, never on the number of threads. */
int leekrl_vec_step(leekrl_vec *vec, const int32_t *actions);
const char *leekrl_vec_error(const leekrl_vec *vec);

#ifdef __cplusplus
}
#endif
#endif
