#include "leekrl/plant_ai.h"

#include <stddef.h>

#include "lw_area.h"
#include "lw_attack.h"
#include "lw_cell.h"
#include "lw_chip.h"
#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_map.h"
#include "lw_pathfinding.h"
#include "lw_state.h"

typedef struct {
    LwFight *fight;
    LwState *state;
    LwEntity *plant;
    LwChip *area;
    LwChip *single;
    int heals;
    LwEntity *entities[LW_STATE_MAX_ENTITIES];
    int n_entities;
} plant_turn;

static int has_effect(LwChip *chip, int type)
{
    return lw_attack_get_effect_parameters_by_type(lw_chip_get_attack(chip), type) != NULL;
}

static int usable(const plant_turn *t, LwChip *chip)
{
    int uses;
    if (!chip || lw_chip_get_cost(chip) > lw_entity_get_tp(t->plant) ||
        lw_state_has_cooldown(t->state, t->plant, chip)) return 0;
    uses = lw_attack_get_max_uses(lw_chip_get_attack(chip));
    return uses == -1 || lw_entity_get_item_uses(t->plant, lw_chip_get_id(chip)) < uses;
}

static int in_range(const plant_turn *t, LwChip *chip, const LwEntity *target)
{
    return lw_map_can_use_attack(lw_state_get_map(t->state), lw_entity_get_cell(t->plant),
                                 lw_entity_get_cell(target), lw_chip_get_attack(chip));
}

/* Distance from the plant, or -1 if the area chip does not reach the target. */
static int area_distance(const plant_turn *t, const LwEntity *target)
{
    LwCell cells[64];
    LwAttack *attack = lw_chip_get_attack(t->area);
    LwCell *origin = lw_entity_get_cell(t->plant);
    LwCell *cell = lw_entity_get_cell(target);
    int i, n = lw_area_get_area(&attack->area, lw_state_get_map(t->state),
                                origin, origin, t->plant, cells, 64);
    for (i = 0; i < n; ++i)
        if (lw_cell_get_id(&cells[i]) == lw_cell_get_id(cell))
            return lw_pathfinding_get_case_distance(origin, cell);
    return -1;
}

static int missing_life(const LwEntity *entity)
{
    return lw_entity_get_total_life(entity) - lw_entity_get_life(entity);
}

/* Living entities of the side the plant's chips are meant for. */
static int eligible(const plant_turn *t, const LwEntity *e)
{
    int ally = lw_entity_get_team(e) == lw_entity_get_team(t->plant);
    return e != t->plant && !lw_entity_is_dead(e) && lw_entity_get_cell(e) &&
           ally == t->heals;
}

/* Eligible entities that would really be affected. */
static int wanted(const plant_turn *t, const LwEntity *e)
{
    if (!eligible(t, e)) return 0;
    return t->heals
        ? missing_life(e) > 0 && !lw_entity_has_state(e, LW_ENTITY_STATE_UNHEALABLE)
        : !lw_entity_has_state(e, LW_ENTITY_STATE_INVINCIBLE);
}

static int life_limit(const plant_turn *t, const LwEntity *e)
{
    return t->heals ? missing_life(e) : lw_entity_get_life(e);
}

/* Estimated life given to or removed from target by one cast of chip. roll is
 * the engine's random draw in [0, 1). Mirrors the engine's effect formulas. */
static double estimate(const plant_turn *t, LwChip *chip, const LwEntity *target,
                       double aoe, int target_count, double roll)
{
    LwAttack *attack = lw_chip_get_attack(chip);
    double total = 0, limit = life_limit(t, target);
    double power = 1 + lw_entity_get_power(t->plant) / 100.0;
    int i, n = lw_attack_get_effects_count(attack);
    for (i = 0; i < n; ++i) {
        const LwEffectParameters *p = &attack->effects[i];
        double value = (lw_effect_parameters_get_value1(p) +
                        roll * lw_effect_parameters_get_value2(p)) * aoe;
        int multiplied = lw_effect_parameters_get_modifiers(p) &
                         LW_MODIFIER_MULTIPLIED_BY_TARGETS;
        int count = multiplied ? target_count : 1;
        int stat;
        switch (lw_effect_parameters_get_id(p)) {
        case LW_EFFECT_TYPE_HEAL:
            value *= (1 + lw_entity_get_wisdom(t->plant) / 100.0) * count;
            break;
        case LW_EFFECT_TYPE_DAMAGE:
            stat = lw_entity_get_strength(t->plant);
            value *= (1 + (stat > 0 ? stat : 0) / 100.0) * power * count;
            value -= value * lw_entity_get_relative_shield(target) / 100.0 +
                     lw_entity_get_absolute_shield(target);
            break;
        case LW_EFFECT_TYPE_POISON:
            stat = lw_entity_get_magic(t->plant);
            value *= (1 + (stat > 0 ? stat : 0) / 100.0) * power *
                     lw_effect_parameters_get_turns(p);
            break;
        default:
            value = 0;
        }
        if (value > 0) total += value;
    }
    return total < limit ? total : limit;
}

static int better(const plant_turn *t, const LwEntity *a, const LwEntity *b)
{
    int x, y;
    if (!b) return 1;
    x = lw_entity_is_summon(a);
    y = lw_entity_is_summon(b);
    if (x != y) return x < y;
    x = t->heals ? -missing_life(a) : lw_entity_get_life(a);
    y = t->heals ? -missing_life(b) : lw_entity_get_life(b);
    if (x != y) return x < y;
    return lw_entity_get_fid(a) < lw_entity_get_fid(b);
}

static LwEntity *best_target(const plant_turn *t, int lethal_only)
{
    LwEntity *best = NULL;
    int i;
    for (i = 0; i < t->n_entities; ++i) {
        LwEntity *e = t->entities[i];
        if (!wanted(t, e) || !in_range(t, t->single, e)) continue;
        if (lethal_only &&
            estimate(t, t->single, e, 1, 1, 0) < lw_entity_get_life(e)) continue;
        if (better(t, e, best)) best = e;
    }
    return best;
}

static double area_score(const plant_turn *t)
{
    double score = 0;
    int i, count = 0, distance;
    /* Every eligible target multiplies the value, even one at full life. */
    for (i = 0; i < t->n_entities; ++i)
        if (eligible(t, t->entities[i]) && area_distance(t, t->entities[i]) >= 0) ++count;
    for (i = 0; i < t->n_entities; ++i) {
        LwEntity *e = t->entities[i];
        double aoe;
        if (!wanted(t, e) || (distance = area_distance(t, e)) < 0) continue;
        aoe = lw_attack_get_power_for_cell(lw_chip_get_attack(t->area),
                                           lw_entity_get_cell(t->plant),
                                           lw_entity_get_cell(e));
        score += estimate(t, t->area, e, aoe, count, 0.5);
    }
    return score;
}

/* Value of spending the current TP on the best target of the single chip. */
static double single_score(const plant_turn *t)
{
    LwEntity *target;
    double score;
    int cost;
    if (!usable(t, t->single) || !(target = best_target(t, 0))) return 0;
    cost = lw_chip_get_cost(t->single);
    score = estimate(t, t->single, target, 1, 1, 0.5) *
            (cost > 0 ? lw_entity_get_tp(t->plant) / cost : 1);
    return score < life_limit(t, target) ? score : life_limit(t, target);
}

static int cast(const plant_turn *t, LwChip *chip, LwCell *cell)
{
    if (lw_fight_use_chip(t->fight, t->plant, cell, chip) > 0) return 1;
    lw_state_set_fatal_error(t->state, LW_STATE_ERROR_PLANT_DISPATCH,
                             "plant AI cast refused by the engine");
    return 0;
}

int leekrl_plant_ai(LwFight *fight, LwEntity *plant, LwEntity *trigger,
                    int64_t ai_id, int turn, void *userdata)
{
    plant_turn t = {0};
    int i, n;
    (void)trigger;
    (void)ai_id;
    (void)turn;
    (void)userdata;
    t.fight = fight;
    t.state = lw_fight_get_state(fight);
    t.plant = plant;
    n = lw_entity_get_chips_count(plant);
    for (i = 0; i < n; ++i) {
        LwChip *chip = lw_entity_get_chip_at(plant, i);
        int heals = has_effect(chip, LW_EFFECT_TYPE_HEAL);
        if (!heals && !has_effect(chip, LW_EFFECT_TYPE_DAMAGE)) continue;
        t.heals = heals;
        if (lw_attack_get_max_range(lw_chip_get_attack(chip)) == 0) t.area = chip;
        else t.single = chip;
    }
    /* TP strictly decreases; the bound only guards against free chips. */
    for (i = 0; i < 16; ++i) {
        LwEntity *target;
        if (lw_entity_is_dead(plant) || lw_state_get_fatal_error(t.state) ||
            lw_state_is_finished(t.state)) return 0;
        t.n_entities = lw_state_get_all_entities(t.state, 0, t.entities,
                                                 LW_STATE_MAX_ENTITIES);
        if (!t.heals && usable(&t, t.single) && (target = best_target(&t, 1)) != NULL) {
            if (!cast(&t, t.single, lw_entity_get_cell(target))) return 0;
        } else if (usable(&t, t.area) && area_score(&t) > single_score(&t)) {
            if (!cast(&t, t.area, lw_entity_get_cell(plant))) return 0;
        } else if (usable(&t, t.single) && (target = best_target(&t, 0)) != NULL) {
            if (!cast(&t, t.single, lw_entity_get_cell(target))) return 0;
        } else {
            return 0;
        }
    }
    return 0;
}
