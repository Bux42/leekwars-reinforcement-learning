#include "leekrl/actions.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

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

struct leekrl_action_catalog {
    size_t count;
    leekrl_action *actions;
};

typedef struct {
    int id;
    leekrl_action_kind kind;
    LwAttack *attack;
} catalog_item;

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

leekrl_action_status leekrl_action_catalog_create(leekrl_action_catalog **out)
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
    for (i = 0; i < nw; ++i) {
        if (!weapons[i]) { status = LEEKRL_ACTION_ENGINE_ERROR; goto done; }
        items[i].id = lw_weapon_get_id(weapons[i]);
        items[i].kind = LEEKRL_ACTION_WEAPON;
        items[i].attack = lw_weapon_get_attack(weapons[i]);
    }
    n = (size_t)nw;
    for (i = 0; i < nc; ++i) {
        if (!chips[i]) { status = LEEKRL_ACTION_ENGINE_ERROR; goto done; }
        if (plant_only(chips[i], chips, nc)) continue;
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
        count += slots;
    }
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

static leekrl_action_status resolve(LwFight *fight, LwEntity *actor,
                                    const leekrl_action *action,
                                    resolved_action *resolved, movement_routes *routes)
{
    LwState *state;
    LwMap *map;
    LwCell *from;
    LwAttack *attack;
    int64_t x, y;
    int uses;
    if (!fight || !actor || !resolved || !valid_descriptor(action))
        return LEEKRL_ACTION_INVALID_ARGUMENT;
    state = lw_fight_get_state(fight);
    if (!state || !lw_state_get_map(state) || !lw_entity_get_cell(actor) ||
        !lw_state_can_act(state, actor)) return LEEKRL_ACTION_INVALID_STATE;
    if (lw_state_get_fatal_error(state)) return LEEKRL_ACTION_ENGINE_ERROR;
    resolved->target = NULL;
    resolved->weapon = NULL;
    resolved->chip = NULL;
    if (action->kind == LEEKRL_ACTION_END_TURN) return LEEKRL_ACTION_OK;

    map = lw_state_get_map(state);
    from = lw_entity_get_cell(actor);
    x = (int64_t)lw_cell_get_x(from) + action->dx;
    y = (int64_t)lw_cell_get_y(from) + action->dy;
    if (x < map->min_x || x > map->max_x || y < map->min_y || y > map->max_y)
        return LEEKRL_ACTION_ILLEGAL;
    resolved->target = lw_map_get_cell_xy(map, (int)x, (int)y);
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
    if (action->kind == LEEKRL_ACTION_WEAPON) {
        int i, equip;
        for (i = 0; i < lw_entity_get_weapons_count(actor); ++i) {
            LwWeapon *weapon = lw_entity_get_weapon_at(actor, i);
            if (weapon && lw_weapon_get_id(weapon) == action->item_id) {
                resolved->weapon = weapon;
                break;
            }
        }
        if (!resolved->weapon || lw_entity_has_state(actor, LW_ENTITY_STATE_AMPUTATED))
            return LEEKRL_ACTION_ILLEGAL;
        equip = lw_entity_get_weapon(actor) != resolved->weapon;
        if ((equip && lw_entity_get_tp(actor) <= 0) ||
            (int64_t)lw_entity_get_tp(actor) - equip < lw_weapon_get_cost(resolved->weapon))
            return LEEKRL_ACTION_ILLEGAL;
        attack = lw_weapon_get_attack(resolved->weapon);
    } else {
        const LwEffectParameters *summon;
        int team;
        resolved->chip = lw_entity_get_chip(actor, action->item_id);
        if (!resolved->chip) return LEEKRL_ACTION_ILLEGAL;
        attack = lw_chip_get_attack(resolved->chip);
        if (!attack) return LEEKRL_ACTION_ENGINE_ERROR;
        summon = lw_attack_get_effect_parameters_by_type(attack, LW_EFFECT_TYPE_SUMMON);
        team = lw_entity_get_team(actor);
        if ((summon || lw_chip_is_team_cooldown(resolved->chip)) &&
            (team < 0 || team >= state->n_teams || !state->teams[team]))
            return LEEKRL_ACTION_ENGINE_ERROR;
        if (((summon || lw_chip_get_cost(resolved->chip) > 0) &&
             lw_chip_get_cost(resolved->chip) > lw_entity_get_tp(actor)) ||
            lw_state_has_cooldown(state, actor, resolved->chip))
            return LEEKRL_ACTION_ILLEGAL;
        if (summon) {
            double template_id = lw_effect_parameters_get_value1(summon);
            if (!lw_cell_available(resolved->target, map) ||
                lw_entity_has_state(actor, LW_ENTITY_STATE_STERILE) ||
                !lw_map_can_use_attack(map, from, resolved->target, attack))
                return LEEKRL_ACTION_ILLEGAL;
            if (lw_team_get_summon_count(state->teams[team]) >= LW_SUMMON_LIMIT)
                return LEEKRL_ACTION_ILLEGAL;
            if (!isfinite(template_id) || template_id < INT_MIN || template_id > INT_MAX ||
                !lw_bulbs_get_invocation_template((int)template_id) ||
                state->n_entities >= LW_STATE_MAX_ENTITIES ||
                state->teams[team]->n_entities >= LW_TEAM_MAX_ENTITIES ||
                state->order.n_leeks >= LW_ORDER_MAX_ENTITIES)
                return LEEKRL_ACTION_ENGINE_ERROR;
            return LEEKRL_ACTION_OK;
        }
        if (!lw_cell_is_walkable(resolved->target) ||
            (lw_attack_get_effect_parameters_by_type(attack, LW_EFFECT_TYPE_TELEPORT) &&
             !lw_cell_available(resolved->target, map)) ||
            lw_attack_blocked_by_density(attack, actor, lw_cell_get_player(resolved->target, map)))
            return LEEKRL_ACTION_ILLEGAL;
    }
    if (!attack) return LEEKRL_ACTION_ENGINE_ERROR;
    uses = lw_attack_get_max_uses(attack);
    if ((uses != -1 && lw_entity_get_item_uses(actor, action->item_id) >= uses) ||
        !lw_map_can_use_attack(map, from, resolved->target, attack))
        return LEEKRL_ACTION_ILLEGAL;
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
