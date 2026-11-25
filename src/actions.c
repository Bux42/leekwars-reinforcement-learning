#include "leekrl/actions.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lw_area.h"
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

typedef struct {
    LwCell *target;
    LwWeapon *weapon;
    LwChip *chip;
    LwCell *path[LEEKRL_MAX_MOVE_DISTANCE];
    int path_length;
} resolved_action;

typedef struct {
    int ready;
    int distance[LW_MAP_MAX_CELLS];
    int parent[LW_MAP_MAX_CELLS];
    int reached[LW_MAP_MAX_CELLS]; /* by distance, the start first */
    int n_reached;
} movement_routes;

/* An item and its actions: actions[first] to actions[first + count - 1]. */
typedef struct {
    int id;
    leekrl_action_kind kind;
    LwAttack *attack;
    size_t first;
    size_t count;
} catalog_item;

struct leekrl_action_catalog {
    size_t count;
    leekrl_action *actions;
    size_t item_count;
    catalog_item *items; /* sorted by compare_items */
    /* Action moving by (dx, dy), as move[dy + MAX][dx + MAX]. 0 for none. */
    size_t move[2 * LEEKRL_MAX_MOVE_DISTANCE + 1][2 * LEEKRL_MAX_MOVE_DISTANCE + 1];
};

/* Target-independent part of casting an item. */
typedef struct {
    LwWeapon *weapon;
    LwChip *chip;
    LwAttack *attack;
    const LwEffectParameters *summon;
    int team;
    int teleports;
    int checks_density;
} castable;

/* Line of sight from one cell, computed on demand: -1 unknown, 0 or 1. */
typedef struct {
    signed char visible[LW_MAP_MAX_CELLS];
} sight_cache;

static int compare_items(const void *a, const void *b)
{
    const catalog_item *x = a, *y = b;
    if (x->kind != y->kind) return x->kind < y->kind ? -1 : 1;
    return (x->id > y->id) - (x->id < y->id);
}

static int template_owns(const LwBulbTemplate *bulb, const LwChip *chip)
{
    int i, n = lw_bulb_template_get_chips_count(bulb);
    for (i = 0; i < n; ++i)
        if (lw_bulb_template_get_chip_at(bulb, i) == chip) return 1;
    return 0;
}

/* Plants are scripted: chips owned by awakening plants and no other summon
 * are not actions. Templates are found through the chips that summon them. */
static int plant_only(const LwChip *chip, LwChip **chips, int count)
{
    int i, plant = 0;
    for (i = 0; i < count; ++i) {
        const LwEffectParameters *summon;
        const LwBulbTemplate *bulb;
        double id;
        if (!chips[i]) continue;
        summon = lw_attack_get_effect_parameters_by_type(lw_chip_get_attack(chips[i]),
                                                         LW_EFFECT_TYPE_SUMMON);
        if (!summon) continue;
        id = lw_effect_parameters_get_value1(summon);
        if (!isfinite(id) || id < INT_MIN || id > INT_MAX) continue;
        bulb = lw_bulbs_get_invocation_template((int)id);
        if (!bulb || !template_owns(bulb, chip)) continue;
        if (bulb->awakening_zone <= 0) return 0;
        plant = 1;
    }
    return plant;
}

static int excluded_item(const int *excluded, size_t count, int id)
{
    size_t i;
    for (i = 0; i < count; ++i)
        if (excluded[i] == id) return 1;
    return 0;
}

leekrl_action_status leekrl_action_catalog_create(leekrl_action_catalog **out)
{
    return leekrl_action_catalog_create_without(NULL, 0, out);
}

leekrl_action_status leekrl_action_catalog_create_without(const int *excluded,
                                                          size_t excluded_count,
                                                          leekrl_action_catalog **out)
{
    LwWeapon **weapons;
    LwChip **chips;
    catalog_item *items = NULL;
    leekrl_action_catalog *catalog = NULL;
    leekrl_action_status status = LEEKRL_ACTION_OK;
    int nw, nc, i;
    size_t n, count = 1 + LEEKRL_MOVE_ACTION_COUNT, offset;
    int dx, dy;
    if (!out) return LEEKRL_ACTION_INVALID_ARGUMENT;
    *out = NULL;
    if (!excluded && excluded_count) return LEEKRL_ACTION_INVALID_ARGUMENT;
    weapons = lw_weapons_get_templates(&nw);
    chips = lw_chips_get_templates(&nc);
    if (nw < 0 || nc < 0 || (nw && !weapons) || (nc && !chips))
        return LEEKRL_ACTION_ENGINE_ERROR;
    n = (size_t)nw + (size_t)nc;
    if (n > SIZE_MAX / sizeof(*items)) return LEEKRL_ACTION_OUT_OF_MEMORY;
    if (n) {
        items = malloc(n * sizeof(*items));
        if (!items) return LEEKRL_ACTION_OUT_OF_MEMORY;
    }
    n = 0;
    for (i = 0; i < nw; ++i) {
        if (!weapons[i]) { status = LEEKRL_ACTION_ENGINE_ERROR; goto done; }
        if (excluded_item(excluded, excluded_count, lw_weapon_get_id(weapons[i]))) continue;
        items[n].id = lw_weapon_get_id(weapons[i]);
        items[n].kind = LEEKRL_ACTION_WEAPON;
        items[n].attack = lw_weapon_get_attack(weapons[i]);
        ++n;
    }
    for (i = 0; i < nc; ++i) {
        if (!chips[i]) { status = LEEKRL_ACTION_ENGINE_ERROR; goto done; }
        if (plant_only(chips[i], chips, nc) ||
            excluded_item(excluded, excluded_count, lw_chip_get_id(chips[i]))) continue;
        items[n].id = lw_chip_get_id(chips[i]);
        items[n].kind = LEEKRL_ACTION_CHIP;
        items[n].attack = lw_chip_get_attack(chips[i]);
        ++n;
    }
    if (n) qsort(items, n, sizeof(*items), compare_items);
    for (offset = 0; offset < n; ++offset) {
        size_t slots;
        status = leekrl_action_cast_layout(items[offset].attack, items[offset].kind,
                                           items[offset].id, NULL, 0, &slots);
        if (status != LEEKRL_ACTION_OK) goto done;
        if (slots > SIZE_MAX - count) { status = LEEKRL_ACTION_OUT_OF_MEMORY; goto done; }
        count += slots;
    }
    if (count > SIZE_MAX / sizeof(leekrl_action)) {
        status = LEEKRL_ACTION_OUT_OF_MEMORY;
        goto done;
    }
    catalog = calloc(1, sizeof(*catalog));
    if (!catalog) { status = LEEKRL_ACTION_OUT_OF_MEMORY; goto done; }
    catalog->actions = calloc(count, sizeof(*catalog->actions));
    if (!catalog->actions) { status = LEEKRL_ACTION_OUT_OF_MEMORY; goto done; }
    catalog->count = count;
    catalog->actions[0] = (leekrl_action){LEEKRL_ACTION_END_TURN, 0, 0, 0};
    catalog->actions[1] = (leekrl_action){LEEKRL_ACTION_MOVE, 0, 0, -1};
    catalog->actions[2] = (leekrl_action){LEEKRL_ACTION_MOVE, 0, -1, 0};
    catalog->actions[3] = (leekrl_action){LEEKRL_ACTION_MOVE, 0, 1, 0};
    catalog->actions[4] = (leekrl_action){LEEKRL_ACTION_MOVE, 0, 0, 1};
    count = 5;
    for (dy = -LEEKRL_MAX_MOVE_DISTANCE; dy <= LEEKRL_MAX_MOVE_DISTANCE; ++dy) {
        for (dx = -LEEKRL_MAX_MOVE_DISTANCE; dx <= LEEKRL_MAX_MOVE_DISTANCE; ++dx) {
            int distance = abs(dx) + abs(dy);
            if (distance < 2 || distance > LEEKRL_MAX_MOVE_DISTANCE) continue;
            catalog->actions[count++] = (leekrl_action){LEEKRL_ACTION_MOVE, 0, dx, dy};
        }
    }
    for (offset = 0; offset < n; ++offset) {
        size_t slots;
        status = leekrl_action_cast_layout(items[offset].attack, items[offset].kind,
                                           items[offset].id, catalog->actions + count,
                                           catalog->count - count, &slots);
        if (status != LEEKRL_ACTION_OK) goto done;
        items[offset].first = count;
        items[offset].count = slots;
        count += slots;
    }
    for (offset = 1; offset <= LEEKRL_MOVE_ACTION_COUNT; ++offset) {
        const leekrl_action *move = &catalog->actions[offset];
        catalog->move[move->dy + LEEKRL_MAX_MOVE_DISTANCE]
                     [move->dx + LEEKRL_MAX_MOVE_DISTANCE] = offset;
    }
    catalog->items = items;
    catalog->item_count = n;
    items = NULL;
    *out = catalog;
    catalog = NULL;
done:
    free(items);
    leekrl_action_catalog_destroy(catalog);
    return status;
}

void leekrl_action_catalog_destroy(leekrl_action_catalog *catalog)
{
    if (!catalog) return;
    free(catalog->actions);
    free(catalog->items);
    free(catalog);
}

size_t leekrl_action_catalog_count(const leekrl_action_catalog *catalog)
{
    return catalog ? catalog->count : 0;
}

const leekrl_action *leekrl_action_catalog_actions(const leekrl_action_catalog *catalog)
{
    return catalog ? catalog->actions : NULL;
}

leekrl_action_status leekrl_action_cast_layout(const LwAttack *attack,
                                               leekrl_action_kind kind,
                                               int item_id,
                                               leekrl_action *actions,
                                               size_t capacity,
                                               size_t *required)
{
    LwCell origin = {0}, target = {0};
    int range, dx, dy, pass;
    size_t count = 0;
    if (!attack || !required || (kind != LEEKRL_ACTION_WEAPON && kind != LEEKRL_ACTION_CHIP) ||
        (!actions && capacity) || lw_attack_get_min_range(attack) < 0 ||
        lw_attack_get_max_range(attack) < lw_attack_get_min_range(attack))
        return LEEKRL_ACTION_INVALID_ARGUMENT;
    range = lw_attack_get_max_range(attack);
    if (range > 2 * (LW_MAP_COORD_DIM - 1)) range = 2 * (LW_MAP_COORD_DIM - 1);
    for (pass = 0; pass < 2; ++pass) {
        count = 0;
        for (dy = -range; dy <= range; ++dy) {
            for (dx = -range; dx <= range; ++dx) {
                if (dx <= -LW_MAP_COORD_DIM || dx >= LW_MAP_COORD_DIM ||
                    dy <= -LW_MAP_COORD_DIM || dy >= LW_MAP_COORD_DIM) continue;
                target.x = dx;
                target.y = dy;
                if (!lw_map_verify_range(NULL, &origin, dx || dy ? &target : &origin, attack))
                    continue;
                if (pass) {
                    actions[count].kind = kind;
                    actions[count].item_id = item_id;
                    actions[count].dx = dx;
                    actions[count].dy = dy;
                }
                ++count;
            }
        }
        if (!pass) {
            *required = count;
            if (!actions && !capacity) return LEEKRL_ACTION_OK;
            if (capacity < count) return LEEKRL_ACTION_BUFFER_TOO_SMALL;
        }
    }
    return LEEKRL_ACTION_OK;
}

static int valid_descriptor(const leekrl_action *action)
{
    if (!action) return 0;
    switch (action->kind) {
    case LEEKRL_ACTION_END_TURN:
    case LEEKRL_ACTION_WEAPON:
    case LEEKRL_ACTION_CHIP:
        return 1;
    case LEEKRL_ACTION_MOVE: {
        int64_t dx = action->dx, dy = action->dy;
        int64_t distance = (dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy);
        return distance > 0 && distance <= LEEKRL_MAX_MOVE_DISTANCE;
    }
    }
    return 0;
}

static int build_movement_routes(LwMap *map, LwEntity *actor, movement_routes *routes)
{
    int *queue = routes->reached, head = 0, tail = 0, i;
    int limit = lw_entity_get_mp(actor);
    LwCell *from = lw_entity_get_cell(actor);
    int start = lw_cell_get_id(from);
    if (map->nb_cells < 1 || map->nb_cells > LW_MAP_MAX_CELLS ||
        start < 0 || start >= map->nb_cells) return 0;
    for (i = 0; i < map->nb_cells; ++i) {
        routes->distance[i] = -1;
        routes->parent[i] = -1;
    }
    routes->ready = 1;
    routes->n_reached = 0;
    if (limit > LEEKRL_MAX_MOVE_DISTANCE) limit = LEEKRL_MAX_MOVE_DISTANCE;
    if (limit <= 0 || lw_entity_has_state(actor, LW_ENTITY_STATE_STATIC) ||
        lw_entity_has_state(actor, LW_ENTITY_STATE_ROOTED)) return 1;
    routes->distance[start] = 0;
    queue[tail++] = start;
    while (head < tail) {
        int id = queue[head++], d;
        LwCell *around[4];
        if (routes->distance[id] == limit) continue;
        lw_map_get_cells_around(map, lw_map_get_cell(map, id), around);
        for (d = 0; d < 4; ++d) {
            int next;
            if (!around[d] || !lw_cell_available(around[d], map)) continue;
            next = lw_cell_get_id(around[d]);
            if (next < 0 || next >= map->nb_cells) return 0;
            if (routes->distance[next] >= 0) continue;
            routes->distance[next] = routes->distance[id] + 1;
            routes->parent[next] = id;
            queue[tail++] = next;
        }
    }
    routes->n_reached = tail;
    return 1;
}

static leekrl_action_status check_actor(LwFight *fight, LwEntity *actor)
{
    LwState *state = lw_fight_get_state(fight);
    if (!state || !lw_state_get_map(state) || !lw_entity_get_cell(actor) ||
        !lw_state_can_act(state, actor)) return LEEKRL_ACTION_INVALID_STATE;
    return lw_state_get_fatal_error(state) ? LEEKRL_ACTION_ENGINE_ERROR : LEEKRL_ACTION_OK;
}

static leekrl_action_status check_item(LwState *state, LwEntity *actor,
                                       leekrl_action_kind kind, int item_id,
                                       castable *item)
{
    int i, equip, uses;
    item->weapon = NULL;
    item->chip = NULL;
    item->attack = NULL;
    item->summon = NULL;
    item->team = lw_entity_get_team(actor);
    item->teleports = 0;
    item->checks_density = 0;
    if (kind == LEEKRL_ACTION_WEAPON) {
        for (i = 0; i < lw_entity_get_weapons_count(actor); ++i) {
            LwWeapon *weapon = lw_entity_get_weapon_at(actor, i);
            if (weapon && lw_weapon_get_id(weapon) == item_id) {
                item->weapon = weapon;
                break;
            }
        }
        if (!item->weapon || lw_entity_has_state(actor, LW_ENTITY_STATE_AMPUTATED))
            return LEEKRL_ACTION_ILLEGAL;
        equip = lw_entity_get_weapon(actor) != item->weapon;
        if ((equip && lw_entity_get_tp(actor) <= 0) ||
            (int64_t)lw_entity_get_tp(actor) - equip < lw_weapon_get_cost(item->weapon))
            return LEEKRL_ACTION_ILLEGAL;
        item->attack = lw_weapon_get_attack(item->weapon);
        if (!item->attack) return LEEKRL_ACTION_ENGINE_ERROR;
    } else {
        item->chip = lw_entity_get_chip(actor, item_id);
        if (!item->chip) return LEEKRL_ACTION_ILLEGAL;
        item->attack = lw_chip_get_attack(item->chip);
        if (!item->attack) return LEEKRL_ACTION_ENGINE_ERROR;
        item->summon = lw_attack_get_effect_parameters_by_type(item->attack,
                                                               LW_EFFECT_TYPE_SUMMON);
        if ((item->summon || lw_chip_is_team_cooldown(item->chip)) &&
            (item->team < 0 || item->team >= state->n_teams || !state->teams[item->team]))
            return LEEKRL_ACTION_ENGINE_ERROR;
        if (((item->summon || lw_chip_get_cost(item->chip) > 0) &&
             lw_chip_get_cost(item->chip) > lw_entity_get_tp(actor)) ||
            lw_state_has_cooldown(state, actor, item->chip))
            return LEEKRL_ACTION_ILLEGAL;
        if (item->summon)
            return lw_entity_has_state(actor, LW_ENTITY_STATE_STERILE) ||
                   lw_team_get_summon_count(state->teams[item->team]) >= LW_SUMMON_LIMIT
                ? LEEKRL_ACTION_ILLEGAL : LEEKRL_ACTION_OK;
        item->teleports = lw_attack_get_effect_parameters_by_type(
            item->attack, LW_EFFECT_TYPE_TELEPORT) != NULL;
        item->checks_density = item->teleports || lw_attack_get_effect_parameters_by_type(
            item->attack, LW_EFFECT_TYPE_PERMUTATION) != NULL;
    }
    uses = lw_attack_get_max_uses(item->attack);
    return uses != -1 && lw_entity_get_item_uses(actor, item_id) >= uses
        ? LEEKRL_ACTION_ILLEGAL : LEEKRL_ACTION_OK;
}

/* sight may be NULL. Otherwise it must belong to this map state and cell, and the
 * target come from the catalog: its range and direction are then known to be right. */
static int can_reach(LwMap *map, LwCell *from, LwCell *target, const LwAttack *attack,
                     sight_cache *sight)
{
    signed char *visible;
    if (!sight && !lw_map_verify_range(map, from, target, attack)) return 0;
    /* First-in-line attacks ignore one entity: their sight is not shared. */
    if (!sight || !lw_attack_need_los(attack) ||
        lw_attack_get_area_id(attack) == LW_AREA_TYPE_FIRST_IN_LINE)
        return lw_map_verify_los(map, from, target, attack);
    visible = &sight->visible[lw_cell_get_id(target)];
    if (*visible < 0) *visible = (signed char)(lw_map_verify_los(map, from, target, attack) != 0);
    return *visible;
}

static leekrl_action_status check_target(LwState *state, LwEntity *actor, const castable *item,
                                         LwCell *target, sight_cache *sight)
{
    LwMap *map = lw_state_get_map(state);
    LwCell *from = lw_entity_get_cell(actor);
    if (item->summon) {
        double template_id = lw_effect_parameters_get_value1(item->summon);
        if (!lw_cell_available(target, map) ||
            !can_reach(map, from, target, item->attack, sight)) return LEEKRL_ACTION_ILLEGAL;
        return !isfinite(template_id) || template_id < INT_MIN || template_id > INT_MAX ||
               !lw_bulbs_get_invocation_template((int)template_id) ||
               state->n_entities >= LW_STATE_MAX_ENTITIES ||
               state->teams[item->team]->n_entities >= LW_TEAM_MAX_ENTITIES ||
               state->order.n_leeks >= LW_ORDER_MAX_ENTITIES
            ? LEEKRL_ACTION_ENGINE_ERROR : LEEKRL_ACTION_OK;
    }
    if (item->chip &&
        (!lw_cell_is_walkable(target) ||
         (item->teleports && !lw_cell_available(target, map)) ||
         (item->checks_density &&
          lw_attack_blocked_by_density(item->attack, actor, lw_cell_get_player(target, map)))))
        return LEEKRL_ACTION_ILLEGAL;
    return can_reach(map, from, target, item->attack, sight)
        ? LEEKRL_ACTION_OK : LEEKRL_ACTION_ILLEGAL;
}

static LwCell *cell_offset(LwMap *map, LwCell *from, int dx, int dy)
{
    int64_t x = (int64_t)lw_cell_get_x(from) + dx;
    int64_t y = (int64_t)lw_cell_get_y(from) + dy;
    if (x < map->min_x || x > map->max_x || y < map->min_y || y > map->max_y) return NULL;
    return lw_map_get_cell_xy(map, (int)x, (int)y);
}

static leekrl_action_status resolve(LwFight *fight, LwEntity *actor,
                                    const leekrl_action *action,
                                    resolved_action *resolved, movement_routes *routes)
{
    LwState *state;
    LwMap *map;
    castable item;
    leekrl_action_status status;
    if (!fight || !actor || !resolved || !valid_descriptor(action))
        return LEEKRL_ACTION_INVALID_ARGUMENT;
    status = check_actor(fight, actor);
    if (status != LEEKRL_ACTION_OK) return status;
    resolved->target = NULL;
    resolved->weapon = NULL;
    resolved->chip = NULL;
    if (action->kind == LEEKRL_ACTION_END_TURN) return LEEKRL_ACTION_OK;

    state = lw_fight_get_state(fight);
    map = lw_state_get_map(state);
    resolved->target = cell_offset(map, lw_entity_get_cell(actor), action->dx, action->dy);
    if (!resolved->target) return LEEKRL_ACTION_ILLEGAL;
    if (action->kind == LEEKRL_ACTION_MOVE) {
        int id = lw_cell_get_id(resolved->target), i;
        if (!routes->ready && !build_movement_routes(map, actor, routes))
            return LEEKRL_ACTION_ENGINE_ERROR;
        if (id < 0 || id >= map->nb_cells) return LEEKRL_ACTION_ENGINE_ERROR;
        resolved->path_length = routes->distance[id];
        if (resolved->path_length <= 0) return LEEKRL_ACTION_ILLEGAL;
        for (i = resolved->path_length - 1; i >= 0; --i) {
            resolved->path[i] = lw_map_get_cell(map, id);
            id = routes->parent[id];
        }
        return LEEKRL_ACTION_OK;
    }
    status = check_item(state, actor, action->kind, action->item_id, &item);
    if (status != LEEKRL_ACTION_OK) return status;
    resolved->weapon = item.weapon;
    resolved->chip = item.chip;
    return check_target(state, actor, &item, resolved->target, NULL);
}

static const catalog_item *find_item(const leekrl_action_catalog *catalog,
                                     leekrl_action_kind kind, int id)
{
    catalog_item key = {0};
    key.id = id;
    key.kind = kind;
    return catalog->item_count ? bsearch(&key, catalog->items, catalog->item_count,
                                         sizeof(key), compare_items) : NULL;
}

/* Passive effects of weapons, without a name in the generator. They act whichever
 * weapon is in hand. */
enum {
    POISON_TO_SCIENCE = 33,
    DAMAGE_TO_ABSOLUTE_SHIELD = 34,
    DAMAGE_TO_STRENGTH = 35,
    NOVA_DAMAGE_TO_MAGIC = 36, /* and the maximum life lost with any damage */
    KILL_TO_TP = 56,
    CRITICAL_TO_HEAL = 58
};
#define PASSIVE(id) (UINT64_C(1) << (id))

static uint64_t passive_effects(const LwEntity *actor)
{
    uint64_t found = 0;
    int i, k;
    for (i = 0; i < lw_entity_get_weapons_count(actor); ++i) {
        const LwWeapon *weapon = lw_entity_get_weapon_at(actor, i);
        for (k = 0; k < lw_weapon_get_passive_effects_count(weapon); ++k) {
            int id = lw_effect_parameters_get_id(&lw_weapon_get_passive_effects(weapon)[k]);
            if (id >= 0 && id < 64) found |= PASSIVE(id);
        }
    }
    return found;
}

/* What makes some casts worth it. */
typedef struct {
    uint64_t passives; /* those of the caster */
    int kills;         /* someone is rewarded for kills */
    int propagation;   /* someone carries a poison that propagates */
    int summon;        /* the caster is one */
    int healable;      /* the caster would receive a heal */
} circumstances;

/* Whether a heal would give it anything now. */
static int healable(const LwEntity *entity)
{
    return lw_entity_get_life(entity) < lw_entity_get_total_life(entity) &&
           !lw_entity_has_state(entity, LW_ENTITY_STATE_UNHEALABLE);
}

/* Entities an effect can be worth reaching. */
enum { ON_CASTER = 1, ON_ALLIES = 2, ON_ENEMIES = 4, ON_ANYONE = 7 };

typedef struct {
    int id;
    int targets;  /* those the engine accepts */
    int worth;    /* ON_... */
    int replaces; /* removes the same effect of the same item from its target */
    int heals;    /* gives life at once, and nothing else */
    int on_caster;
    int kept_out; /* by the same item's effects its target already has */
} effect_use;

/* What a cast needs to be kept. */
typedef struct {
    int always;     /* whatever is reached */
    int any_entity; /* an entity of any kind is enough */
    int caster_healable;
    int item;
    int n_effects;
    effect_use effects[LW_ATTACK_MAX_EFFECTS];
} usefulness;

/* Only judges effects of items that were looked at one by one. */
static int worth_reaching(const LwAttack *attack, const LwEffectParameters *effect,
                          const circumstances *now)
{
    uint64_t fed; /* passive effects that make damage to the caster worth something */
    switch (lw_effect_parameters_get_id(effect)) {
    case LW_EFFECT_TYPE_SHACKLE_MP:
    case LW_EFFECT_TYPE_SHACKLE_TP:
    case LW_EFFECT_TYPE_SHACKLE_STRENGTH:
    case LW_EFFECT_TYPE_SHACKLE_AGILITY:
    case LW_EFFECT_TYPE_SHACKLE_WISDOM:
        return ON_ENEMIES;
    case LW_EFFECT_TYPE_ADD_STATE:
        return (int)lw_effect_parameters_get_value1(effect) == LW_ENTITY_STATE_UNHEALABLE
            ? ON_ENEMIES : ON_ANYONE;
    case LW_EFFECT_TYPE_HEAL:
    case LW_EFFECT_TYPE_VITALITY:
    case LW_EFFECT_TYPE_NOVA_VITALITY:
        /* Life given to an enemy can change which poison kills it. */
        return now->kills ? ON_ANYONE : ON_CASTER | ON_ALLIES;
    case LW_EFFECT_TYPE_RAW_BUFF_MAGIC:
        /* A poison that propagates can come back to the enemy that cast it, with the
         * magic it then has. */
        return now->propagation ? ON_ANYONE : ON_CASTER | ON_ALLIES;
    case LW_EFFECT_TYPE_RAW_BUFF_TP:
    case LW_EFFECT_TYPE_RAW_BUFF_STRENGTH:
    case LW_EFFECT_TYPE_RAW_BUFF_SCIENCE:
    case LW_EFFECT_TYPE_RAW_BUFF_AGILITY:
    case LW_EFFECT_TYPE_RAW_BUFF_RESISTANCE:
    case LW_EFFECT_TYPE_RAW_BUFF_WISDOM:
    case LW_EFFECT_TYPE_REMOVE_SHACKLES:
        return ON_CASTER | ON_ALLIES;
    case LW_EFFECT_TYPE_POISON:
        /* From the caster to the enemies around. */
        if (lw_attack_get_effect_parameters_by_type(attack, LW_EFFECT_TYPE_PROPAGATION))
            return ON_ANYONE;
        fed = PASSIVE(POISON_TO_SCIENCE) | PASSIVE(NOVA_DAMAGE_TO_MAGIC);
        break;
    case LW_EFFECT_TYPE_NOVA_DAMAGE:
        fed = PASSIVE(NOVA_DAMAGE_TO_MAGIC);
        break;
    case LW_EFFECT_TYPE_DAMAGE:
    case LW_EFFECT_TYPE_LIFE_DAMAGE:
    case LW_EFFECT_TYPE_VULNERABILITY:
    case LW_EFFECT_TYPE_ABSOLUTE_VULNERABILITY:
        fed = PASSIVE(DAMAGE_TO_ABSOLUTE_SHIELD) | PASSIVE(DAMAGE_TO_STRENGTH) |
              PASSIVE(LW_EFFECT_TYPE_DAMAGE_TO_RESISTANCE) | PASSIVE(NOVA_DAMAGE_TO_MAGIC);
        break;
    default:
        return ON_ANYONE;
    }
    /* A summon may want to die: it frees a place. */
    return (now->passives & fed) || now->summon ? ON_ANYONE : ON_ALLIES | ON_ENEMIES;
}

static void examine(const LwAttack *attack, leekrl_trim trim, const circumstances *now,
                    usefulness *use)
{
    int i;
    use->always = use->any_entity = use->n_effects = 0;
    use->item = lw_attack_get_item_id(attack);
    use->caster_healable = now->healable;
    for (i = 0; i < lw_attack_get_effects_count(attack); ++i) {
        const LwEffectParameters *effect = &attack->effects[i], *reached = effect;
        int id = lw_effect_parameters_get_id(effect);
        int modifiers = lw_effect_parameters_get_modifiers(effect);
        int worth = trim >= LEEKRL_TRIM_HARMFUL
            ? worth_reaching(attack, effect, now) : ON_ANYONE;
        int heals = (id == LW_EFFECT_TYPE_HEAL || id == LW_EFFECT_TYPE_RAW_HEAL) &&
            lw_effect_parameters_get_turns(effect) == 0;
        effect_use *added;
        switch (id) {
        case LW_EFFECT_TYPE_SUMMON:
        case LW_EFFECT_TYPE_RESURRECT:
        case LW_EFFECT_TYPE_TELEPORT:
            use->always = 1;
            return;
        case LW_EFFECT_TYPE_ATTRACT:
        case LW_EFFECT_TYPE_PUSH:
        case LW_EFFECT_TYPE_REPEL:
            use->any_entity = 1;
            continue;
        case LW_EFFECT_TYPE_PROPAGATION: /* only modifies the other effects */
            continue;
        case LW_EFFECT_TYPE_STEAL_ABSOLUTE_SHIELD:
        case LW_EFFECT_TYPE_STEAL_LIFE:
            /* Gives what the previous effect did: nothing if it reached nobody. */
            if (!i) continue;
            reached = &attack->effects[i - 1];
            worth = ON_ANYONE;
            break;
        default:
            if (!(modifiers & LW_MODIFIER_ON_CASTER)) break;
            if (modifiers & LW_MODIFIER_MULTIPLIED_BY_TARGETS) {
                worth = ON_ANYONE; /* each target counts for the caster */
            } else if ((worth & ON_CASTER) && (!heals || now->healable)) {
                use->always = 1;
                return;
            } else {
                continue;
            }
        }
        added = &use->effects[use->n_effects++];
        added->id = id;
        added->targets = lw_effect_parameters_get_targets(reached);
        added->worth = worth;
        added->heals = heals;
        added->on_caster = (modifiers & LW_MODIFIER_ON_CASTER) != 0;
        added->kept_out = (modifiers & LW_MODIFIER_NOT_REPLACEABLE) != 0;
        added->replaces = lw_effect_parameters_get_turns(effect) != 0 &&
            (!(modifiers & LW_MODIFIER_STACKABLE) || id == LW_EFFECT_TYPE_ADD_STATE);
    }
}

static int has_effect(const LwEntity *entity, int id, int item)
{
    int i;
    for (i = 0; i < lw_entity_get_effects_count(entity); ++i) {
        const LwEffect *effect = lw_entity_get_effect_at(entity, i);
        const LwAttack *attack = lw_effect_get_attack(effect);
        if (lw_effect_get_id(effect) == id && attack && lw_attack_get_item_id(attack) == item)
            return 1;
    }
    return 0;
}

static int has_poison(const LwEntity *entity, int permanent_too)
{
    int i;
    for (i = 0; i < lw_entity_get_effects_count(entity); ++i) {
        const LwEffect *effect = lw_entity_get_effect_at(entity, i);
        if (lw_effect_get_id(effect) == LW_EFFECT_TYPE_POISON &&
            (permanent_too || lw_effect_get_turns(effect) != -1)) return 1;
    }
    return 0;
}

static int has_reducible_effect(const LwEntity *entity)
{
    int i;
    for (i = 0; i < lw_entity_get_effects_count(entity); ++i)
        if (!(lw_effect_get_modifiers(lw_entity_get_effect_at(entity, i)) &
              LW_MODIFIER_IRREDUCTIBLE)) return 1;
    return 0;
}

/* Whether an effect can change anything on an entity the engine accepts. */
static int receives(const usefulness *use, const effect_use *effect, const LwEntity *entity)
{
    if (effect->heals) return effect->on_caster ? use->caster_healable : healable(entity);
    if (effect->kept_out && lw_entity_has_effect(entity, use->item)) return 0;
    switch (effect->id) {
    case LW_EFFECT_TYPE_NOVA_DAMAGE: /* at most the life missing */
        return lw_entity_get_life(entity) != lw_entity_get_total_life(entity);
    case LW_EFFECT_TYPE_SUPERINFECTION:
        return has_poison(entity, 0);
    case LW_EFFECT_TYPE_ANTIDOTE:
        return has_poison(entity, 1);
    case LW_EFFECT_TYPE_DEBUFF:
        return has_reducible_effect(entity);
    default:
        return 1;
    }
}

static int accepts(const usefulness *use, LwEntity *actor, LwEntity *entity)
{
    int side = entity == actor ? ON_CASTER
        : lw_entity_get_team(entity) == lw_entity_get_team(actor) ? ON_ALLIES : ON_ENEMIES;
    int i;
    if (use->any_entity) return 1;
    for (i = 0; i < use->n_effects; ++i) {
        const effect_use *effect = &use->effects[i];
        if (!lw_attack_filter_target(effect->targets, actor, entity) ||
            !receives(use, effect, entity)) continue;
        if ((effect->worth & side) ||
            (effect->replaces && has_effect(entity, effect->id, use->item))) return 1;
    }
    return 0;
}

/* Largest distance of a cell of a fixed-shape area to its center. */
static int area_reach(int area)
{
    switch (area) {
    case LW_AREA_TYPE_SINGLE_CELL: return 0;
    case LW_AREA_TYPE_CIRCLE1: return 1;
    case LW_AREA_TYPE_CIRCLE2: case LW_AREA_TYPE_AREA_PLUS_2: case LW_AREA_TYPE_X_1:
    case LW_AREA_TYPE_SQUARE_1: return 2;
    case LW_AREA_TYPE_CIRCLE3: case LW_AREA_TYPE_AREA_PLUS_3: return 3;
    case LW_AREA_TYPE_X_2: case LW_AREA_TYPE_SQUARE_2: return 4;
    case LW_AREA_TYPE_X_3: return 6;
    default: return -1; /* lines, allies, enemies */
    }
}

/* Whether a fixed-shape area holds the cell at (dx, dy) from its center. */
static int in_area(int area, int dx, int dy)
{
    int reach = area_reach(area);
    dx = abs(dx);
    dy = abs(dy);
    if (dx + dy > reach) return 0;
    switch (area) {
    case LW_AREA_TYPE_AREA_PLUS_2: case LW_AREA_TYPE_AREA_PLUS_3: return !dx || !dy;
    case LW_AREA_TYPE_X_1: case LW_AREA_TYPE_X_2: case LW_AREA_TYPE_X_3: return dx == dy;
    case LW_AREA_TYPE_SQUARE_1: case LW_AREA_TYPE_SQUARE_2:
        return dx <= reach / 2 && dy <= reach / 2;
    default: return 1;
    }
}

/* Entities an item is worth reaching, as far as its range and area allow. */
typedef struct {
    int n;
    LwCell *cells[LW_ORDER_MAX_ENTITIES];
} candidates;

static void find_candidates(LwState *state, LwEntity *actor, const LwAttack *attack,
                            const usefulness *use, candidates *found)
{
    LwCell *from = lw_entity_get_cell(actor);
    int reach = area_reach(lw_attack_get_area_id(attack)), i;
    int limit = lw_attack_get_max_range(attack) + reach;
    found->n = 0;
    for (i = 0; i < state->order.n_leeks; ++i) {
        LwEntity *entity = state->order.leeks[i];
        LwCell *cell = lw_entity_get_cell(entity);
        if (!cell || lw_entity_is_dead(entity) ||
            (reach >= 0 && lw_pathfinding_get_case_distance(from, cell) > limit) ||
            !accepts(use, actor, entity)) continue;
        found->cells[found->n++] = cell;
    }
}

/* Whether a cast has a candidate in its area. */
static int useful(LwState *state, LwEntity *actor, const LwAttack *attack,
                  const usefulness *use, const candidates *found, LwCell *target)
{
    LwMap *map = lw_state_get_map(state);
    int area = lw_attack_get_area_id(attack), i, n;
    LwCell cells[64];
    if (area_reach(area) >= 0) {
        for (i = 0; i < found->n; ++i)
            if (in_area(area, found->cells[i]->x - target->x, found->cells[i]->y - target->y))
                return 1;
        return 0;
    }
    n = lw_area_get_area(&attack->area, map, lw_entity_get_cell(actor), target, actor, cells, 64);
    for (i = 0; i < n; ++i) {
        LwEntity *entity = lw_cell_get_player(&cells[i], map);
        if (entity && !lw_entity_is_dead(entity) && accepts(use, actor, entity)) return 1;
    }
    return 0;
}

/* Position in the catalog of the cast of an item at (dx, dy). Layouts are sorted by dy,
 * then dx. Returns 0, the end of turn, if there is none. */
static size_t find_slot(const leekrl_action_catalog *catalog, const catalog_item *item,
                        int dx, int dy)
{
    size_t low = item->first, high = item->first + item->count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        const leekrl_action *slot = &catalog->actions[middle];
        if (slot->dy < dy || (slot->dy == dy && slot->dx < dx)) low = middle + 1;
        else high = middle;
    }
    return low < item->first + item->count && catalog->actions[low].dx == dx &&
           catalog->actions[low].dy == dy ? low : 0;
}

size_t leekrl_action_catalog_find(const leekrl_action_catalog *catalog,
                                  const leekrl_action *action)
{
    const catalog_item *item;
    size_t found = 0;
    if (!catalog || !action) return 0;
    switch (action->kind) {
    case LEEKRL_ACTION_END_TURN:
        return 0;
    case LEEKRL_ACTION_MOVE:
        if (abs(action->dx) <= LEEKRL_MAX_MOVE_DISTANCE &&
            abs(action->dy) <= LEEKRL_MAX_MOVE_DISTANCE)
            found = catalog->move[action->dy + LEEKRL_MAX_MOVE_DISTANCE]
                                 [action->dx + LEEKRL_MAX_MOVE_DISTANCE];
        break;
    default:
        item = find_item(catalog, action->kind, action->item_id);
        if (item) found = find_slot(catalog, item, action->dx, action->dy);
    }
    return found ? found : catalog->count;
}

static leekrl_action_status mask_item(const leekrl_action_catalog *catalog, LwState *state,
                                      LwEntity *actor, leekrl_action_kind kind, int id,
                                      sight_cache *sight, leekrl_trim trim,
                                      const circumstances *now, uint8_t *mask)
{
    const catalog_item *entry = find_item(catalog, kind, id);
    LwMap *map = lw_state_get_map(state);
    LwCell *from = lw_entity_get_cell(actor);
    leekrl_action_status status;
    castable item;
    usefulness use;
    candidates found;
    size_t i;
    if (!entry) return LEEKRL_ACTION_OK;
    status = check_item(state, actor, kind, id, &item);
    if (status == LEEKRL_ACTION_ILLEGAL) return LEEKRL_ACTION_OK;
    if (status != LEEKRL_ACTION_OK) return status;
    if (trim) {
        examine(item.attack, trim, now, &use);
        if (use.always) trim = LEEKRL_TRIM_NOTHING;
    }
    if (trim) {
        find_candidates(state, actor, item.attack, &use, &found);
        if (!found.n) return LEEKRL_ACTION_OK;
        if (lw_attack_get_area_id(item.attack) == LW_AREA_TYPE_SINGLE_CELL) {
            int k;
            for (k = 0; k < found.n; ++k) {
                LwCell *target = found.cells[k];
                i = find_slot(catalog, entry, target->x - from->x, target->y - from->y);
                if (!i) continue;
                status = check_target(state, actor, &item, target, sight);
                if (status == LEEKRL_ACTION_OK) mask[i] = 1;
                else if (status != LEEKRL_ACTION_ILLEGAL) return status;
            }
            return LEEKRL_ACTION_OK;
        }
    }
    for (i = entry->first; i < entry->first + entry->count; ++i) {
        LwCell *target = cell_offset(map, from, catalog->actions[i].dx, catalog->actions[i].dy);
        if (!target) continue;
        /* Much cheaper than the line of sight. */
        if (trim && !useful(state, actor, item.attack, &use, &found, target)) continue;
        status = check_target(state, actor, &item, target, sight);
        if (status == LEEKRL_ACTION_OK) mask[i] = 1;
        else if (status != LEEKRL_ACTION_ILLEGAL) return status;
    }
    return LEEKRL_ACTION_OK;
}

static leekrl_action_status catalog_mask(const leekrl_action_catalog *catalog,
                                         LwFight *fight, LwEntity *actor,
                                         leekrl_trim trim, uint8_t *mask)
{
    LwState *state;
    LwMap *map;
    LwCell *from;
    movement_routes routes = {0};
    sight_cache sight;
    leekrl_action_status status;
    circumstances now = {0};
    int i;
    if (!catalog || !fight || !actor || !mask || (unsigned)trim > LEEKRL_TRIM_HARMFUL)
        return LEEKRL_ACTION_INVALID_ARGUMENT;
    status = check_actor(fight, actor);
    if (status != LEEKRL_ACTION_OK) return status;
    state = lw_fight_get_state(fight);
    map = lw_state_get_map(state);
    from = lw_entity_get_cell(actor);
    memset(mask, 0, catalog->count);
    mask[0] = 1;
    if (!build_movement_routes(map, actor, &routes)) return LEEKRL_ACTION_ENGINE_ERROR;
    for (i = 1; i < routes.n_reached; ++i) {
        const LwCell *cell = &map->cells[routes.reached[i]];
        mask[catalog->move[cell->y - from->y + LEEKRL_MAX_MOVE_DISTANCE]
                          [cell->x - from->x + LEEKRL_MAX_MOVE_DISTANCE]] = 1;
    }
    memset(sight.visible, -1, sizeof(sight.visible));
    now.passives = passive_effects(actor);
    now.summon = lw_entity_is_summon(actor);
    now.healable = healable(actor);
    for (i = 0; i < state->order.n_leeks; ++i) {
        const LwEntity *entity = state->order.leeks[i];
        int k;
        now.kills |= (passive_effects(entity) & PASSIVE(KILL_TO_TP)) != 0;
        for (k = 0; k < lw_entity_get_effects_count(entity); ++k)
            now.propagation |= lw_entity_get_effect_at(entity, k)->propagate > 0;
    }
    /* Casts of this entity can matter by themselves, through critical hits. The heal comes
     * before anything else the cast does. */
    if ((now.passives & PASSIVE(CRITICAL_TO_HEAL)) && now.healable &&
        lw_entity_get_agility(actor) > 0) trim = LEEKRL_TRIM_NOTHING;
    for (i = 0; i < lw_entity_get_weapons_count(actor); ++i) {
        LwWeapon *weapon = lw_entity_get_weapon_at(actor, i);
        if (!weapon) continue;
        status = mask_item(catalog, state, actor, LEEKRL_ACTION_WEAPON,
                           lw_weapon_get_id(weapon), &sight, trim, &now, mask);
        if (status != LEEKRL_ACTION_OK) return status;
    }
    for (i = 0; i < lw_entity_get_chips_count(actor); ++i) {
        LwChip *chip = lw_entity_get_chip_at(actor, i);
        if (!chip) continue;
        status = mask_item(catalog, state, actor, LEEKRL_ACTION_CHIP,
                           lw_chip_get_id(chip), &sight, trim, &now, mask);
        if (status != LEEKRL_ACTION_OK) return status;
    }
    return LEEKRL_ACTION_OK;
}

leekrl_action_status leekrl_action_catalog_mask(const leekrl_action_catalog *catalog,
                                                LwFight *fight, LwEntity *actor,
                                                uint8_t *mask)
{
    return catalog_mask(catalog, fight, actor, LEEKRL_TRIM_NOTHING, mask);
}

leekrl_action_status leekrl_action_catalog_trimmed_mask(const leekrl_action_catalog *catalog,
                                                        LwFight *fight, LwEntity *actor,
                                                        leekrl_trim trim, uint8_t *mask)
{
    return catalog_mask(catalog, fight, actor, trim, mask);
}

leekrl_action_status leekrl_action_mask(LwFight *fight, LwEntity *actor,
                                        const leekrl_action *actions,
                                        size_t count, uint8_t *mask)
{
    size_t i;
    movement_routes routes = {0};
    if (!fight || !actor || !actions || !count || !mask)
        return LEEKRL_ACTION_INVALID_ARGUMENT;
    for (i = 0; i < count; ++i) {
        resolved_action resolved;
        leekrl_action_status status = resolve(fight, actor, &actions[i], &resolved, &routes);
        if (status != LEEKRL_ACTION_OK && status != LEEKRL_ACTION_ILLEGAL)
            return status;
        mask[i] = status == LEEKRL_ACTION_OK ? 1 : 0;
    }
    return LEEKRL_ACTION_OK;
}

leekrl_action_status leekrl_action_execute(LwFight *fight, LwEntity *actor,
                                           const leekrl_action *action,
                                           int *engine_result)
{
    resolved_action resolved;
    leekrl_action_status status;
    LwState *state;
    movement_routes routes = {0};
    int result;
    if (!engine_result) return LEEKRL_ACTION_INVALID_ARGUMENT;
    status = resolve(fight, actor, action, &resolved, &routes);
    if (status != LEEKRL_ACTION_OK) return status;
    state = lw_fight_get_state(fight);
    switch (action->kind) {
    case LEEKRL_ACTION_END_TURN:
        *engine_result = 0;
        return LEEKRL_ACTION_OK;
    case LEEKRL_ACTION_MOVE:
        result = lw_state_move_entity_path(state, actor, resolved.path, resolved.path_length);
        break;
    case LEEKRL_ACTION_WEAPON:
        if (lw_entity_get_weapon(actor) != resolved.weapon &&
            !lw_state_set_weapon(state, actor, resolved.weapon))
            return LEEKRL_ACTION_ENGINE_ERROR;
        result = lw_state_use_weapon(state, actor, resolved.target);
        break;
    case LEEKRL_ACTION_CHIP:
        if (lw_attack_get_effect_parameters_by_type(lw_chip_get_attack(resolved.chip),
                                                    LW_EFFECT_TYPE_SUMMON))
            result = lw_fight_summon_entity_with_ai_id(fight, actor, resolved.target,
                                                       resolved.chip, LEEKRL_SUMMON_AI_ID);
        else
            result = lw_fight_use_chip(fight, actor, resolved.target, resolved.chip);
        break;
    default:
        return LEEKRL_ACTION_INVALID_ARGUMENT;
    }
    *engine_result = result;
    if (result <= 0 || lw_state_get_fatal_error(state)) return LEEKRL_ACTION_ENGINE_ERROR;
    return LEEKRL_ACTION_OK;
}

const char *leekrl_action_status_string(leekrl_action_status status)
{
    switch (status) {
    case LEEKRL_ACTION_OK: return "ok";
    case LEEKRL_ACTION_INVALID_ARGUMENT: return "invalid argument";
    case LEEKRL_ACTION_INVALID_STATE: return "actor cannot act";
    case LEEKRL_ACTION_ILLEGAL: return "illegal action";
    case LEEKRL_ACTION_ENGINE_ERROR: return "generator error";
    case LEEKRL_ACTION_BUFFER_TOO_SMALL: return "action buffer too small";
    case LEEKRL_ACTION_OUT_OF_MEMORY: return "out of memory";
    }
    return "unknown action status";
}
