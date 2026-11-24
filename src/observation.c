#include "leekrl/observation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lw_attack.h"
#include "lw_bulb_template.h"
#include "lw_chip.h"
#include "lw_effect.h"
#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_map.h"
#include "lw_pathfinding.h"
#include "lw_state.h"
#include "lw_team.h"
#include "lw_weapon.h"

#define SIDE (2 * LEEKRL_OBS_MAP_RADIUS + 1)
#define MAX_AMOUNT 5.0f
#define MAX_DURATION 0.9f
#define PERMANENT 1.0f

typedef struct {
    int id;
    float scale;
} effect_slot;

/* Sorted by id: see specs/Observation.md for the groups of scales. */
static const effect_slot effect_slots[LEEKRL_OBS_EFFECTS] = {
    {2, 500},   /* heal over time */
    {3, 500},   /* strength */
    {4, 500},   /* agility */
    {5, 100},   /* relative shield */
    {6, 500},   /* absolute shield */
    {7, 10},    /* MP */
    {8, 10},    /* TP */
    {13, 500},  /* poison */
    {17, 10},   /* MP shackle */
    {18, 10},   /* TP shackle */
    {19, 500},  /* strength shackle */
    {20, 100},  /* damage return */
    {21, 500},  /* resistance */
    {22, 500},  /* wisdom */
    {25, 500},  /* aftereffect */
    {26, 100},  /* vulnerability */
    {27, 500},  /* absolute vulnerability */
    {29, 500},  /* stolen absolute shield */
    {31, 10},   /* raw MP */
    {32, 10},   /* raw TP */
    {37, 500},  /* raw absolute shield */
    {38, 500},  /* raw strength */
    {39, 500},  /* raw magic */
    {40, 500},  /* raw science */
    {41, 500},  /* raw agility */
    {42, 500},  /* raw resistance */
    {44, 500},  /* raw wisdom */
    {47, 500},  /* agility shackle */
    {48, 500},  /* wisdom shackle */
    {52, 100},  /* raw power */
    {54, 100}   /* raw relative shield */
};

static const int state_slots[LEEKRL_OBS_STATES] = {
    LW_ENTITY_STATE_UNHEALABLE, LW_ENTITY_STATE_DENSE, LW_ENTITY_STATE_ROOTED,
    LW_ENTITY_STATE_STERILE, LW_ENTITY_STATE_AMPUTATED
};

struct leekrl_observer {
    /* Position by item id, or -1. */
    int *weapon_index;
    int *chip_index;
    int *limited_index;
    size_t ids;
    size_t weapons;
    size_t chips;
    size_t limited;
    size_t floats;
    short map_index[SIDE][SIDE]; /* [dy][dx], -1 outside the radius */
};

/* ---- Layout ---------------------------------------------------------- */

int leekrl_obs_effect_index(int effect_id)
{
    int low = 0, high = LEEKRL_OBS_EFFECTS - 1;
    while (low <= high) {
        int middle = (low + high) / 2;
        if (effect_slots[middle].id == effect_id) return middle;
        if (effect_slots[middle].id < effect_id) low = middle + 1;
        else high = middle - 1;
    }
    return -1;
}

int leekrl_obs_state_index(int state_id)
{
    int i;
    for (i = 0; i < LEEKRL_OBS_STATES; ++i)
        if (state_slots[i] == state_id) return i;
    return -1;
}

int leekrl_obs_map_index(int dx, int dy)
{
    int ady = abs(dy), radius = LEEKRL_OBS_MAP_RADIUS, before;
    if (abs(dx) > radius || ady > radius || abs(dx) + ady > radius) return -1;
    /* Rows above hold 1, 3, 5... cells, then shrink again below the center. */
    if (dy <= 0) before = (dy + radius) * (dy + radius);
    else before = LEEKRL_OBS_MAP_CELLS - (radius - dy + 1) * (radius - dy + 1);
    return before + dx + (radius - ady);
}

static int lookup(const leekrl_observer *observer, const int *table, int id)
{
    return observer && id >= 0 && (size_t)id < observer->ids ? table[id] : -1;
}

int leekrl_observer_weapon_index(const leekrl_observer *observer, int item_id)
{
    return lookup(observer, observer ? observer->weapon_index : NULL, item_id);
}

int leekrl_observer_chip_index(const leekrl_observer *observer, int item_id)
{
    return lookup(observer, observer ? observer->chip_index : NULL, item_id);
}

int leekrl_observer_limited_index(const leekrl_observer *observer, int item_id)
{
    return lookup(observer, observer ? observer->limited_index : NULL, item_id);
}

size_t leekrl_observer_weapons(const leekrl_observer *observer) { return observer ? observer->weapons : 0; }
size_t leekrl_observer_chips(const leekrl_observer *observer) { return observer ? observer->chips : 0; }
size_t leekrl_observer_limited(const leekrl_observer *observer) { return observer ? observer->limited : 0; }
size_t leekrl_observer_float_count(const leekrl_observer *observer) { return observer ? observer->floats : 0; }

leekrl_observer *leekrl_observer_create(const leekrl_action_catalog *catalog)
{
    const leekrl_action *actions;
    leekrl_observer *observer;
    size_t i, n;
    int max_id = 0, dx, dy;
    if (!catalog) return NULL;
    actions = leekrl_action_catalog_actions(catalog);
    n = leekrl_action_catalog_count(catalog);
    for (i = 0; i < n; ++i) {
        if (actions[i].item_id < 0) return NULL;
        if (actions[i].item_id > max_id) max_id = actions[i].item_id;
    }
    observer = calloc(1, sizeof(*observer));
    if (!observer) return NULL;
    observer->ids = (size_t)max_id + 1;
    observer->weapon_index = malloc(observer->ids * sizeof(int));
    observer->chip_index = malloc(observer->ids * sizeof(int));
    observer->limited_index = malloc(observer->ids * sizeof(int));
    if (!observer->weapon_index || !observer->chip_index || !observer->limited_index) {
        leekrl_observer_destroy(observer);
        return NULL;
    }
    for (i = 0; i < observer->ids; ++i)
        observer->weapon_index[i] = observer->chip_index[i] = observer->limited_index[i] = -1;
    /* Catalog items are sorted: weapons by id, then chips by id. */
    for (i = 0; i < n; ++i) {
        int id = actions[i].item_id, uses;
        if (actions[i].kind == LEEKRL_ACTION_WEAPON) {
            const LwWeapon *weapon = lw_weapons_get_weapon(id);
            if (observer->weapon_index[id] >= 0) continue;
            if (!weapon) {
                leekrl_observer_destroy(observer);
                return NULL;
            }
            observer->weapon_index[id] = (int)observer->weapons++;
            uses = lw_attack_get_max_uses(lw_weapon_get_attack((LwWeapon *)weapon));
        } else if (actions[i].kind == LEEKRL_ACTION_CHIP) {
            const LwChip *chip = lw_chips_get_chip(id);
            if (observer->chip_index[id] >= 0) continue;
            if (!chip) {
                leekrl_observer_destroy(observer);
                return NULL;
            }
            observer->chip_index[id] = (int)observer->chips++;
            uses = lw_attack_get_max_uses(lw_chip_get_attack((LwChip *)chip));
        } else {
            continue;
        }
        if (uses != -1) observer->limited_index[id] = (int)observer->limited++;
    }
    observer->floats = LEEKRL_OBS_ITEMS +
                       2 * (2 * observer->weapons + 2 * observer->chips) + observer->limited;
    for (dy = -LEEKRL_OBS_MAP_RADIUS; dy <= LEEKRL_OBS_MAP_RADIUS; ++dy)
        for (dx = -LEEKRL_OBS_MAP_RADIUS; dx <= LEEKRL_OBS_MAP_RADIUS; ++dx)
            observer->map_index[dy + LEEKRL_OBS_MAP_RADIUS][dx + LEEKRL_OBS_MAP_RADIUS] =
                (short)leekrl_obs_map_index(dx, dy);
    return observer;
}

void leekrl_observer_destroy(leekrl_observer *observer)
{
    if (!observer) return;
    free(observer->weapon_index);
    free(observer->chip_index);
    free(observer->limited_index);
    free(observer);
}

/* ---- Writing --------------------------------------------------------- */

static int fail(char *error, size_t size, const char *message, int value)
{
    if (error && size) snprintf(error, size, "%s (%d)", message, value);
    return 0;
}

static float duration(int turns)
{
    float value;
    if (turns == -1) return PERMANENT;
    value = (float)turns / 10;
    return value < MAX_DURATION ? value : MAX_DURATION;
}

static float cooldown(int turns)
{
    float value = (float)turns / 10;
    return value < 1 ? value : 1;
}

static int write_effects(LwEntity *entity, float *out, char *error, size_t error_size)
{
    int i, n = lw_entity_get_effects_count(entity);
    for (i = 0; i < n; ++i) {
        const LwEffect *effect = lw_entity_get_effect_at(entity, i);
        float turns = duration(effect->turns), *slot;
        int index;
        if (effect->id == LW_EFFECT_TYPE_ADD_STATE) {
            index = leekrl_obs_state_index(effect->state);
            if (index < 0) return fail(error, error_size, "unlisted state", effect->state);
            slot = out + LEEKRL_OBS_STATE + 2 * index;
            slot[0] = 1;
        } else {
            index = leekrl_obs_effect_index(effect->id);
            if (index < 0) return fail(error, error_size, "unlisted effect", effect->id);
            slot = out + LEEKRL_OBS_EFFECT + 2 * index;
            slot[0] += (float)effect->value / effect_slots[index].scale;
        }
        if (turns > slot[1]) slot[1] = turns;
    }
    for (i = 0; i < LEEKRL_OBS_EFFECTS; ++i)
        if (out[LEEKRL_OBS_EFFECT + 2 * i] > MAX_AMOUNT)
            out[LEEKRL_OBS_EFFECT + 2 * i] = MAX_AMOUNT;
    return 1;
}

static int write_entity(LwState *state, LwEntity *actor, LwEntity *entity, float *out,
                        char *error, size_t error_size)
{
    LwCell *from = lw_entity_get_cell(actor), *cell = lw_entity_get_cell(entity);
    int summon = lw_entity_is_summon(entity), type = 0, i;
    int total_life = lw_entity_get_total_life(entity);
    int zone = entity->awakening_zone;
    if (!cell) return fail(error, error_size, "entity without a cell", entity->fight_id);
    out[LEEKRL_OBS_PRESENT] = 1;
    out[LEEKRL_OBS_ALLY] = lw_entity_get_team(entity) == lw_entity_get_team(actor);
    if (summon) {
        const LwBulbTemplate *bulb;
        type = lw_entity_get_skin(entity); /* the id of its template */
        bulb = type >= 1 && type <= LEEKRL_OBS_SUMMON_TYPES
            ? lw_bulbs_get_invocation_template(type) : NULL;
        if (!bulb || lw_bulb_template_get_chips_count(bulb) > LEEKRL_OBS_SUMMON_CHIPS)
            return fail(error, error_size, "unlisted summon type", type);
        for (i = 0; i < lw_bulb_template_get_chips_count(bulb); ++i)
            out[LEEKRL_OBS_SUMMON_COOLDOWN + i] = cooldown(lw_state_get_cooldown(
                state, entity, lw_bulb_template_get_chip_at(bulb, i)));
    }
    out[LEEKRL_OBS_TYPE + type] = 1;
    out[LEEKRL_OBS_LIFE] = total_life > 0
        ? (float)lw_entity_get_life(entity) / (float)total_life : 0;
    out[LEEKRL_OBS_TOTAL_LIFE] = (float)(log(1.0 + (total_life > 0 ? total_life : 0)) / 10);
    out[LEEKRL_OBS_DX] = (float)(lw_cell_get_x(cell) - lw_cell_get_x(from)) /
                         LEEKRL_OBS_MAP_RADIUS;
    out[LEEKRL_OBS_DY] = (float)(lw_cell_get_y(cell) - lw_cell_get_y(from)) /
                         LEEKRL_OBS_MAP_RADIUS;
    out[LEEKRL_OBS_DISTANCE] = (float)lw_pathfinding_get_case_distance(from, cell) /
                               LEEKRL_OBS_MAP_RADIUS;
    out[LEEKRL_OBS_TOTAL_TP] = (float)lw_entity_get_total_tp(entity) / 30;
    out[LEEKRL_OBS_TOTAL_MP] = (float)lw_entity_get_total_mp(entity) / 15;
    /* An awakening plant recovers its TP each time it awakens. */
    out[LEEKRL_OBS_TP] = zone > 0 ? out[LEEKRL_OBS_TOTAL_TP]
                                  : (float)lw_entity_get_tp(entity) / 30;
    out[LEEKRL_OBS_MP] = (float)lw_entity_get_mp(entity) / 15;
    out[LEEKRL_OBS_STRENGTH] = (float)lw_entity_get_strength(entity) / 1000;
    out[LEEKRL_OBS_WISDOM] = (float)lw_entity_get_wisdom(entity) / 1000;
    out[LEEKRL_OBS_AGILITY] = (float)lw_entity_get_agility(entity) / 1000;
    out[LEEKRL_OBS_RESISTANCE] = (float)lw_entity_get_resistance(entity) / 1000;
    out[LEEKRL_OBS_SCIENCE] = (float)lw_entity_get_science(entity) / 1000;
    out[LEEKRL_OBS_MAGIC] = (float)lw_entity_get_magic(entity) / 1000;
    out[LEEKRL_OBS_FREQUENCY] = (float)lw_entity_get_frequency(entity) / 1000;
    out[LEEKRL_OBS_POWER] = (float)lw_entity_get_power(entity) / 100;
    out[LEEKRL_OBS_DAMAGE_RETURN] = (float)lw_entity_get_damage_return(entity) / 100;
    out[LEEKRL_OBS_ABSOLUTE_SHIELD] = (float)lw_entity_get_absolute_shield(entity) / 500;
    out[LEEKRL_OBS_RELATIVE_SHIELD] = (float)lw_entity_get_relative_shield(entity) / 100;
    if (zone > 0) {
        out[LEEKRL_OBS_ZONE] = (float)zone / 10;
        for (i = 0; i < entity->n_awakened_by; ++i)
            if (entity->awakened_by[i] == lw_entity_get_fid(actor)) out[LEEKRL_OBS_AWAKENED] = 1;
    }
    return write_effects(entity, out, error, error_size);
}

/* Equipped items, chip cooldowns, weapon in hand. */
static int write_items(const leekrl_observer *observer, LwState *state, LwEntity *leek,
                       float *out, char *error, size_t error_size)
{
    float *cooldowns = out + observer->weapons + observer->chips;
    float *in_hand = cooldowns + observer->chips;
    LwWeapon *held = lw_entity_get_weapon(leek);
    int i, index;
    for (i = 0; i < lw_entity_get_weapons_count(leek); ++i) {
        int id = lw_weapon_get_id(lw_entity_get_weapon_at(leek, i));
        index = leekrl_observer_weapon_index(observer, id);
        if (index < 0) return fail(error, error_size, "unlisted weapon", id);
        out[index] = 1;
    }
    for (i = 0; i < lw_entity_get_chips_count(leek); ++i) {
        LwChip *chip = lw_entity_get_chip_at(leek, i);
        index = leekrl_observer_chip_index(observer, lw_chip_get_id(chip));
        if (index < 0) return fail(error, error_size, "unlisted chip", lw_chip_get_id(chip));
        out[observer->weapons + (size_t)index] = 1;
        cooldowns[index] = cooldown(lw_state_get_cooldown(state, leek, chip));
    }
    if (held) {
        index = leekrl_observer_weapon_index(observer, lw_weapon_get_id(held));
        if (index < 0) return fail(error, error_size, "unlisted weapon", lw_weapon_get_id(held));
        in_hand[index] = 1;
    }
    return 1;
}

static void write_uses(const leekrl_observer *observer, LwEntity *actor, float *out)
{
    int i;
    for (i = 0; i < lw_entity_get_weapons_count(actor); ++i) {
        LwWeapon *weapon = lw_entity_get_weapon_at(actor, i);
        int id = lw_weapon_get_id(weapon), index = leekrl_observer_limited_index(observer, id);
        if (index >= 0)
            out[index] = (float)(lw_attack_get_max_uses(lw_weapon_get_attack(weapon)) -
                                 lw_entity_get_item_uses(actor, id)) / 5;
    }
    for (i = 0; i < lw_entity_get_chips_count(actor); ++i) {
        LwChip *chip = lw_entity_get_chip_at(actor, i);
        int id = lw_chip_get_id(chip), index = leekrl_observer_limited_index(observer, id);
        if (index >= 0)
            out[index] = (float)(lw_attack_get_max_uses(lw_chip_get_attack(chip)) -
                                 lw_entity_get_item_uses(actor, id)) / 5;
    }
}

static int write_map(const leekrl_observer *observer, const LwMap *map, const LwCell *from,
                      uint8_t *out)
{
    int i;
    memset(out, LEEKRL_CELL_OUTSIDE, LEEKRL_OBS_MAP_CELLS);
    for (i = 0; i < map->nb_cells; ++i) {
        const LwCell *cell = &map->cells[i];
        int index = observer->map_index[cell->y - from->y + LEEKRL_OBS_MAP_RADIUS]
                                       [cell->x - from->x + LEEKRL_OBS_MAP_RADIUS];
        if (index < 0) return 0;
        out[index] = lw_cell_is_walkable(cell) ? LEEKRL_CELL_FREE : LEEKRL_CELL_OBSTACLE;
    }
    return 1;
}

static uint8_t category(const LwEntity *actor, const LwEntity *entity)
{
    int summon = lw_entity_is_summon(entity);
    if (entity == actor) return LEEKRL_CELL_SELF;
    if (lw_entity_get_team(entity) == lw_entity_get_team(actor))
        return summon ? LEEKRL_CELL_ALLY_SUMMON : LEEKRL_CELL_ALLY_LEEK;
    return summon ? LEEKRL_CELL_ENEMY_SUMMON : LEEKRL_CELL_ENEMY_LEEK;
}

int leekrl_observe(const leekrl_observer *observer, LwFight *fight, LwEntity *actor,
                   float *floats, uint8_t *map, char *error, size_t error_size)
{
    LwState *state;
    LwMap *cells;
    LwCell *from;
    LwEntity *leeks[2] = {NULL, NULL};
    int summons[2] = {0, 0}, position = -1, alive = 0, i, n;
    size_t block, k;
    if (error && error_size) error[0] = 0;
    if (!observer || !fight || !actor || !floats || !map)
        return fail(error, error_size, "missing argument", 0);
    state = lw_fight_get_state(fight);
    cells = state ? lw_state_get_map(state) : NULL;
    from = lw_entity_get_cell(actor);
    if (!cells || !from || lw_entity_is_dead(actor))
        return fail(error, error_size, "actor cannot observe", actor->fight_id);
    n = state->order.n_leeks;
    if (n > LEEKRL_OBS_MAX_ENTITIES) return fail(error, error_size, "too many entities", n);
    if (cells->max_x - cells->min_x > LEEKRL_OBS_MAP_RADIUS ||
        cells->max_y - cells->min_y > LEEKRL_OBS_MAP_RADIUS)
        return fail(error, error_size, "map too large", cells->nb_cells);
    for (i = 0; i < n; ++i)
        if (state->order.leeks[i] == actor) position = i;
    if (position < 0) return fail(error, error_size, "actor not in the turn order", actor->fight_id);
    for (i = 0; i < state->n_entities; ++i) alive += !lw_entity_is_dead(state->m_entities[i]);
    if (alive != n) return fail(error, error_size, "living entity not in the turn order", alive - n);

    memset(floats, 0, observer->floats * sizeof(*floats));
    if (!write_map(observer, cells, from, map))
        return fail(error, error_size, "map too large", cells->nb_cells);
    for (i = 0; i < n; ++i) {
        LwEntity *entity = state->order.leeks[(position + i) % n];
        LwCell *cell = lw_entity_get_cell(entity);
        int enemy = lw_entity_get_team(entity) != lw_entity_get_team(actor);
        if (!write_entity(state, actor, entity, floats + LEEKRL_OBS_ENTITIES +
                          i * LEEKRL_OBS_ENTITY_SIZE, error, error_size)) return 0;
        map[observer->map_index[cell->y - from->y + LEEKRL_OBS_MAP_RADIUS]
                               [cell->x - from->x + LEEKRL_OBS_MAP_RADIUS]] =
            category(actor, entity);
        if (lw_entity_is_summon(entity)) {
            ++summons[enemy];
        } else {
            if (leeks[enemy]) return fail(error, error_size, "more than one leek per team", enemy);
            leeks[enemy] = entity;
        }
    }
    floats[LEEKRL_OBS_TURN] = (float)lw_state_get_turn(state) / 64;
    floats[LEEKRL_OBS_ALLY_SUMMONS] = (float)summons[0] / 8;
    floats[LEEKRL_OBS_ENEMY_SUMMONS] = (float)summons[1] / 8;
    block = 2 * observer->weapons + 2 * observer->chips;
    for (i = 0; i < 2; ++i) {
        if (!leeks[i]) return fail(error, error_size, "team without a living leek", i);
        if (!write_items(observer, state, leeks[i], floats + LEEKRL_OBS_ITEMS + (size_t)i * block,
                         error, error_size)) return 0;
    }
    write_uses(observer, actor, floats + LEEKRL_OBS_ITEMS + 2 * block);
    for (k = 0; k < observer->floats; ++k)
        if (!isfinite(floats[k])) return fail(error, error_size, "non-finite value at", (int)k);
    return 1;
}
