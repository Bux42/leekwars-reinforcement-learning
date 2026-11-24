#include "leekrl/reward.h"

#include <math.h>
#include <string.h>

#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_state.h"

void leekrl_reward_options_init(leekrl_reward_options *options)
{
    if (!options) return;
    options->win = 1;
    options->loss = -1;
    options->draw = -0.2f;
    options->duration = 0.2f;
    options->actions = 0.1f;
    options->action_scale = 100;
    options->life = 0.3f;
    options->total_life = 0.1f;
    options->summon = 0.01f;
    options->summon_life = 0.01f;
}

int leekrl_reward_options_valid(const leekrl_reward_options *options)
{
    return options && isfinite(options->win) && isfinite(options->loss) &&
           isfinite(options->draw) && isfinite(options->duration) &&
           isfinite(options->actions) && isfinite(options->action_scale) &&
           options->action_scale > 0 && isfinite(options->life) &&
           isfinite(options->total_life) && isfinite(options->summon) &&
           isfinite(options->summon_life);
}

/* Score differences seen by the team, part by part. */
static void compare(const leekrl_reward_tracker *tracker, const leekrl_reward_options *options,
                    LwFight *fight, int team, double difference[LEEKRL_REWARD_OUTCOME])
{
    const LwState *state = lw_fight_get_state(fight);
    int i;
    for (i = 0; i < LEEKRL_REWARD_OUTCOME; ++i) difference[i] = 0;
    for (i = 0; i < state->n_entities; ++i) {
        const LwEntity *e = state->m_entities[i];
        int side = lw_entity_get_team(e);
        double sign = side == team ? 1 : -1;
        double life = lw_entity_get_life(e), total = lw_entity_get_total_life(e);
        if (lw_entity_is_dead(e) || side < 0 || side > 1) continue;
        if (lw_entity_is_summon(e)) {
            difference[LEEKRL_REWARD_SUMMONS] +=
                sign * (options->summon + options->summon_life * (total > 0 ? life / total : 0));
        } else if (tracker->start_life[side] > 0) {
            difference[LEEKRL_REWARD_LIFE] += sign * options->life * life / tracker->start_life[side];
            difference[LEEKRL_REWARD_TOTAL_LIFE] +=
                sign * options->total_life * total / tracker->start_life[side];
        }
    }
}

void leekrl_reward_start(leekrl_reward_tracker *tracker, const leekrl_reward_options *options,
                         LwFight *fight)
{
    const LwState *state = lw_fight_get_state(fight);
    int i;
    memset(tracker, 0, sizeof(*tracker));
    for (i = 0; i < state->n_entities; ++i) {
        const LwEntity *e = state->m_entities[i];
        int side = lw_entity_get_team(e);
        if (!lw_entity_is_summon(e) && (side == 0 || side == 1))
            tracker->start_life[side] = lw_entity_get_total_life(e);
    }
    for (i = 0; i < 2; ++i) compare(tracker, options, fight, i, tracker->last[i]);
}

void leekrl_reward_count_action(leekrl_reward_tracker *tracker, int team)
{
    if (team == 0 || team == 1) ++tracker->actions[team];
}

void leekrl_reward_step(leekrl_reward_tracker *tracker, const leekrl_reward_options *options,
                        LwFight *fight, int team, float parts[LEEKRL_REWARD_PARTS])
{
    double now[LEEKRL_REWARD_OUTCOME];
    int i;
    for (i = 0; i < LEEKRL_REWARD_PARTS; ++i) parts[i] = 0;
    if (team != 0 && team != 1) return;
    compare(tracker, options, fight, team, now);
    for (i = 0; i < LEEKRL_REWARD_OUTCOME; ++i) {
        parts[i] = (float)(now[i] - tracker->last[team][i]);
        tracker->last[team][i] = now[i];
    }
}

void leekrl_reward_end(const leekrl_reward_tracker *tracker,
                       const leekrl_reward_options *options, LwFight *fight, int team,
                       float parts[LEEKRL_REWARD_PARTS])
{
    int winner = lw_fight_get_winner(fight), turn = lw_fight_get_turn(fight);
    if (team != 0 && team != 1) return;
    if (winner != 0 && winner != 1) {
        parts[LEEKRL_REWARD_OUTCOME] += options->draw;
        return;
    }
    if (winner != team) {
        parts[LEEKRL_REWARD_OUTCOME] += options->loss;
        return;
    }
    parts[LEEKRL_REWARD_OUTCOME] += options->win;
    if (turn < LW_FIGHT_MAX_TURNS)
        parts[LEEKRL_REWARD_DURATION] +=
            options->duration * (1 - (float)turn / LW_FIGHT_MAX_TURNS);
    parts[LEEKRL_REWARD_ACTIONS] +=
        options->actions * options->action_scale /
        (options->action_scale + (float)tracker->actions[team]);
}

float leekrl_reward_sum(const float parts[LEEKRL_REWARD_PARTS])
{
    float sum = 0;
    int i;
    for (i = 0; i < LEEKRL_REWARD_PARTS; ++i) sum += parts[i];
    return sum;
}
