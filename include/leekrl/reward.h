#ifndef LEEKRL_REWARD_H
#define LEEKRL_REWARD_H

#ifdef __cplusplus
extern "C" {
#endif

struct LwFight;

/* 1v1 rewards, as specified in specs/Rewards.md. */
typedef struct {
    float win;            /* end of the fight */
    float loss;
    float draw;
    float duration;       /* winner: times (1 - turn / 64) */
    float actions;        /* winner: times action_scale / (action_scale + actions) */
    float action_scale;   /* positive */
    float life;           /* score: leek's life / total life at the start */
    float total_life;     /* score: leek's total life / total life at the start */
    float summon;         /* score: each living summon */
    float summon_life;    /* score: each living summon, times life / total life */
} leekrl_reward_options;

void leekrl_reward_options_init(leekrl_reward_options *options);
int leekrl_reward_options_valid(const leekrl_reward_options *options);

/* Parts of a reward. Their sum is the reward. */
typedef enum {
    LEEKRL_REWARD_LIFE,
    LEEKRL_REWARD_TOTAL_LIFE,
    LEEKRL_REWARD_SUMMONS,
    LEEKRL_REWARD_OUTCOME,
    LEEKRL_REWARD_DURATION,
    LEEKRL_REWARD_ACTIONS,
    LEEKRL_REWARD_PARTS
} leekrl_reward_part;

/* Follows one fight, for both teams. */
typedef struct {
    int start_life[2];
    long actions[2];
    double last[2][LEEKRL_REWARD_OUTCOME]; /* score differences last rewarded */
} leekrl_reward_tracker;

/* Call once the fight has started. */
void leekrl_reward_start(leekrl_reward_tracker *tracker, const leekrl_reward_options *options,
                         struct LwFight *fight);

/* Count a movement or a cast of any entity of the team. Not end of turn. */
void leekrl_reward_count_action(leekrl_reward_tracker *tracker, int team);

/*
 * Fill parts with what changed for the team since the previous call for it, or
 * since the start. Call at each decision of the team and once when the fight is
 * over, BEFORE lw_fight_step_finish, which removes the summons.
 */
void leekrl_reward_step(leekrl_reward_tracker *tracker, const leekrl_reward_options *options,
                        struct LwFight *fight, int team, float parts[LEEKRL_REWARD_PARTS]);

/* Add the end of the fight to parts. Call AFTER lw_fight_step_finish. */
void leekrl_reward_end(const leekrl_reward_tracker *tracker,
                       const leekrl_reward_options *options, struct LwFight *fight, int team,
                       float parts[LEEKRL_REWARD_PARTS]);

float leekrl_reward_sum(const float parts[LEEKRL_REWARD_PARTS]);

#ifdef __cplusplus
}
#endif
#endif
