#ifndef LEEKRL_PLANT_AI_H
#define LEEKRL_PLANT_AI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct LwEntity;
struct LwFight;

/*
 * Scripted AI for awakening plants, for lw_fight_set_plant_ai_dispatch or
 * lw_generator_set_plant_ai_dispatch. userdata is unused. Deterministic: it
 * draws no randomness besides the engine's own when casting.
 *
 * A plant is expected to own one area chip cast on itself (range 0) and one
 * single-target chip, both dealing damage or both healing. Until it has
 * nothing useful to do, it plays, in this order:
 *
 *   Damage: 1. the single chip on an enemy its minimum damage kills
 *           2. the area chip, if its estimated total damage beats spending
 *              the current TP on the single chip
 *           3. the single chip on the best enemy
 *   Heal:   1. the area chip, if its estimated total heal beats spending the
 *              current TP on the single chip
 *           2. the single chip on the best ally
 *
 * Estimates use the average roll and cap at life, or missing life. Best
 * target: leeks before summons, then lowest life (damage) or most missing
 * life (heal), then lowest fight ID. Invincible enemies and unhealable allies
 * are ignored.
 *
 * Returns 0. An unexpected engine refusal sets the state's fatal error.
 */
int leekrl_plant_ai(struct LwFight *fight, struct LwEntity *plant,
                    struct LwEntity *trigger, int64_t ai_id, int turn,
                    void *userdata);

#ifdef __cplusplus
}
#endif
#endif
