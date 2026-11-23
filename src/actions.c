#include "leekrl/actions.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lw_attack.h"
#include "lw_bulb_template.h"
#include "lw_chip.h"
#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_map.h"
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
    int queue[LW_MAP_MAX_CELLS], head = 0, tail = 0, i;
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

/* sight may be NULL. Otherwise it must belong to this map state and cell. */
static int can_reach(LwMap *map, LwCell *from, LwCell *target, const LwAttack *attack,
                     sight_cache *sight)
{
    signed char *visible;
    if (!lw_map_verify_range(map, from, target, attack)) return 0;
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

static leekrl_action_status mask_item(const leekrl_action_catalog *catalog, LwState *state,
                                      LwEntity *actor, leekrl_action_kind kind, int id,
                                      sight_cache *sight, uint8_t *mask)
{
    const catalog_item *entry = find_item(catalog, kind, id);
    LwMap *map = lw_state_get_map(state);
    LwCell *from = lw_entity_get_cell(actor);
    leekrl_action_status status;
    castable item;
    size_t i;
    if (!entry) return LEEKRL_ACTION_OK;
    status = check_item(state, actor, kind, id, &item);
    if (status == LEEKRL_ACTION_ILLEGAL) return LEEKRL_ACTION_OK;
    if (status != LEEKRL_ACTION_OK) return status;
    for (i = entry->first; i < entry->first + entry->count; ++i) {
        LwCell *target = cell_offset(map, from, catalog->actions[i].dx, catalog->actions[i].dy);
        if (!target) continue;
        status = check_target(state, actor, &item, target, sight);
        if (status == LEEKRL_ACTION_OK) mask[i] = 1;
        else if (status != LEEKRL_ACTION_ILLEGAL) return status;
    }
    return LEEKRL_ACTION_OK;
}

leekrl_action_status leekrl_action_catalog_mask(const leekrl_action_catalog *catalog,
                                                LwFight *fight, LwEntity *actor,
                                                uint8_t *mask)
{
    LwState *state;
    LwMap *map;
    LwCell *from;
    movement_routes routes = {0};
    sight_cache sight;
    leekrl_action_status status;
    int i;
    if (!catalog || !fight || !actor || !mask) return LEEKRL_ACTION_INVALID_ARGUMENT;
    status = check_actor(fight, actor);
    if (status != LEEKRL_ACTION_OK) return status;
    state = lw_fight_get_state(fight);
    map = lw_state_get_map(state);
    from = lw_entity_get_cell(actor);
    memset(mask, 0, catalog->count);
    mask[0] = 1;
    if (!build_movement_routes(map, actor, &routes)) return LEEKRL_ACTION_ENGINE_ERROR;
    for (i = 0; i < map->nb_cells; ++i) {
        const LwCell *cell = &map->cells[i];
        if (routes.distance[i] <= 0) continue;
        mask[catalog->move[cell->y - from->y + LEEKRL_MAX_MOVE_DISTANCE]
                          [cell->x - from->x + LEEKRL_MAX_MOVE_DISTANCE]] = 1;
    }
    memset(sight.visible, -1, sizeof(sight.visible));
    for (i = 0; i < lw_entity_get_weapons_count(actor); ++i) {
        LwWeapon *weapon = lw_entity_get_weapon_at(actor, i);
        if (!weapon) continue;
        status = mask_item(catalog, state, actor, LEEKRL_ACTION_WEAPON,
                           lw_weapon_get_id(weapon), &sight, mask);
        if (status != LEEKRL_ACTION_OK) return status;
    }
    for (i = 0; i < lw_entity_get_chips_count(actor); ++i) {
        LwChip *chip = lw_entity_get_chip_at(actor, i);
        if (!chip) continue;
        status = mask_item(catalog, state, actor, LEEKRL_ACTION_CHIP,
                           lw_chip_get_id(chip), &sight, mask);
        if (status != LEEKRL_ACTION_OK) return status;
    }
    return LEEKRL_ACTION_OK;
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
