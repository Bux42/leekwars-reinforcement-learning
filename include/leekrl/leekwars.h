#ifndef LEEKRL_LEEKWARS_H
#define LEEKRL_LEEKWARS_H

#include "leekrl/env.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct leekrl_leekwars leekrl_leekwars;

enum {
    LEEKRL_LW_WAIT, LEEKRL_LW_APPROACH, LEEKRL_LW_FIRE,
    LEEKRL_LW_ACTION_COUNT
};
enum {
    LEEKRL_LW_TURN, LEEKRL_LW_SELF_LIFE, LEEKRL_LW_ENEMY_LIFE,
    LEEKRL_LW_SELF_CELL, LEEKRL_LW_ENEMY_CELL, LEEKRL_LW_TP,
    LEEKRL_LW_MP, LEEKRL_LW_OBSERVATION_SIZE
};

/*
 * A deliberately small 1v1 scenario: one action per learner AI turn,
 * against an idle opponent. FIRE equips the supplied weapon if needed
 * (the engine charges 1 TP) and fires once; APPROACH uses all available MP.
 * The weapon must already be in the generator's global registry and stay
 * unchanged and registered for this object's lifetime. Only single-cell,
 * enemy-targeting damage weapons with no passives and cost <= 9 TP are
 * supported. max_turns is 1..64.
 *
 * This is deterministic replay, not an incremental engine: reset and each
 * step run the whole generator fight, replaying previous choices. No external
 * AI resolver, registry mutation, or concurrent generator use is supported.
 * Seeds use their low 31 bits, as the generator's seed interface is int.
 * The generator does not load weapon data itself; callers register a weapon
 * first. Observations are raw engine values, not the complete map/state;
 * cells are -1 if unavailable.
 * Wins/losses reward +1/-1; a turn-limit draw rewards zero and terminates
 * the finite-horizon scenario, including when max_turns is less than 64.
 * This adapter does not emit truncations. Final observations describe the
 * engine's post-fight state and must not be bootstrapped; final masks are
 * all zero. Replay cost grows with the episode length.
 *
 * Create env with leekrl_leekwars_callbacks() and context, then destroy env
 * before context. Generator headers and lwengine are build dependencies.
 */
leekrl_env_status leekrl_leekwars_create(int weapon_id, int max_turns,
                                        leekrl_leekwars **out_context);
void leekrl_leekwars_destroy(leekrl_leekwars *context);
leekrl_env_callbacks leekrl_leekwars_callbacks(void);

#ifdef __cplusplus
}
#endif
#endif
