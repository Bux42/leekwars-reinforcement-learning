#include "leekrl/leekwars.h"

#include <stdlib.h>
#include <string.h>

#include "lw_entity.h"
#include "lw_generator.h"
#include "lw_map.h"
#include "lw_state.h"
#include "lw_weapon.h"

struct leekrl_leekwars {
    int weapon_id;
    int max_turns;
    int seed;
    size_t count;
    unsigned char actions[LW_FIGHT_MAX_TURNS];
    LwWeapon *weapon;
};

typedef struct {
    leekrl_leekwars *context;
    size_t seen;
    int captured;
    int failed;
    float *observation;
    uint8_t *mask;
} replay;

static LwEntity *team_entity(LwState *state, int team)
{
    int i, count;
    LwEntity **entities = lw_state_get_entities(state, &count);
    for (i = 0; i < count; ++i) {
        LwEntity *e = entities[i];
        if (lw_entity_get_team(e) == team) return e;
    }
    return NULL;
}

static void observe(LwState *state, float *o)
{
    LwEntity *a = team_entity(state, 0), *b = team_entity(state, 1);
    LwCell *ac = a ? lw_entity_get_cell(a) : NULL;
    LwCell *bc = b ? lw_entity_get_cell(b) : NULL;
    o[LEEKRL_LW_TURN] = (float)lw_state_get_turn(state);
    o[LEEKRL_LW_SELF_LIFE] = a ? (float)lw_entity_get_life(a) : 0;
    o[LEEKRL_LW_ENEMY_LIFE] = b ? (float)lw_entity_get_life(b) : 0;
    o[LEEKRL_LW_SELF_CELL] = ac ? (float)lw_cell_get_id(ac) : -1;
    o[LEEKRL_LW_ENEMY_CELL] = bc ? (float)lw_cell_get_id(bc) : -1;
    o[LEEKRL_LW_TP] = a ? (float)lw_entity_get_tp(a) : 0;
    o[LEEKRL_LW_MP] = a ? (float)lw_entity_get_mp(a) : 0;
}

static void legal(LwState *state, LwEntity *a, LwEntity *b,
                  LwWeapon *weapon, uint8_t *mask)
{
    LwCell *path[64];
    LwCell *from = lw_entity_get_cell(a), *to = lw_entity_get_cell(b);
    int equip = lw_entity_get_weapon(a) != weapon;
    int uses = lw_attack_get_max_uses(lw_weapon_get_attack(weapon));
    memset(mask, 0, LEEKRL_LW_ACTION_COUNT);
    mask[LEEKRL_LW_WAIT] = 1;
    if (!from || !to || lw_entity_is_dead(b) || !lw_state_can_act(state, a))
        return;
    if (lw_entity_get_mp(a) > 0 &&
        !lw_entity_has_state(a, LW_ENTITY_STATE_STATIC) &&
        !lw_entity_has_state(a, LW_ENTITY_STATE_ROOTED) &&
        lw_map_get_path_between(lw_state_get_map(state), from, to, path, 64) > 0)
        mask[LEEKRL_LW_APPROACH] = 1;
    if (!lw_entity_has_state(a, LW_ENTITY_STATE_AMPUTATED) &&
        lw_entity_get_tp(a) >= equip &&
        lw_entity_get_tp(a) - equip >= lw_weapon_get_cost(weapon) &&
        (!equip || lw_entity_get_tp(a) > 0) &&
        (uses == -1 || lw_entity_get_item_uses(a, lw_weapon_get_id(weapon)) < uses) &&
        lw_map_can_use_attack(lw_state_get_map(state), from, to,
                              lw_weapon_get_attack(weapon)))
        mask[LEEKRL_LW_FIRE] = 1;
}

static int dispatch(LwFight *fight, LwEntity *entity, void *ai_file,
                    int turn, void *userdata)
{
    replay *r = userdata;
    LwState *state = lw_fight_get_state(fight);
    LwEntity *enemy;
    uint8_t mask[LEEKRL_LW_ACTION_COUNT];
    size_t action;
    (void)ai_file;
    (void)turn;
    if (lw_entity_get_team(entity) != 0) return 0;
    enemy = team_entity(state, 1);
    if (!enemy) { r->failed = 1; return 0; }
    if (r->seen == r->context->count && !r->captured) {
        observe(state, r->observation);
        legal(state, entity, enemy, r->context->weapon, r->mask);
        r->captured = 1;
    }
    if (r->seen >= r->context->count) return 0;
    action = r->context->actions[r->seen++];
    legal(state, entity, enemy, r->context->weapon, mask);
    if (action >= LEEKRL_LW_ACTION_COUNT || !mask[action]) {
        r->failed = 1;
        return 0;
    }
    if (action == LEEKRL_LW_APPROACH) {
        if (lw_state_move_toward(state, entity, lw_entity_get_fid(enemy), -1) <= 0)
            r->failed = 1;
    } else if (action == LEEKRL_LW_FIRE) {
        if (lw_entity_get_weapon(entity) != r->context->weapon &&
            !lw_state_set_weapon(state, entity, r->context->weapon))
            r->failed = 1;
        if (!r->failed &&
            lw_state_use_weapon(state, entity, lw_entity_get_cell(enemy)) <= 0)
            r->failed = 1;
    }
    return 0;
}

static int run(leekrl_leekwars *c, float *observation, uint8_t *mask,
               int *finished, float *reward)
{
    LwScenario *scenario = calloc(1, sizeof(*scenario));
    LwState *state = calloc(1, sizeof(*state));
    LwFight *fight = calloc(1, sizeof(*fight));
    LwOutcome *outcome = calloc(1, sizeof(*outcome));
    LwGenerator generator;
    LwEntityInfo info;
    replay r;
    int i, rc = -1;
    if (!scenario || !state || !fight || !outcome) goto cleanup;
    if (lw_weapons_get_weapon(c->weapon_id) != c->weapon) goto cleanup;
    lw_scenario_init(scenario);
    scenario->seed = c->seed;
    scenario->max_turns = c->max_turns;
    for (i = 0; i < 2; ++i) {
        lw_entity_info_init(&info);
        info.id = i + 1;
        info.life = 100;
        info.tp = 10;
        info.mp = 3;
        info.level = 1;
        info.weapons[0] = c->weapon_id;
        info.n_weapons = 1;
        if (!lw_scenario_add_entity(scenario, i, &info)) goto cleanup_scenario;
    }
    memset(&r, 0, sizeof(r));
    r.context = c;
    r.observation = observation;
    r.mask = mask;
    lw_generator_init(&generator);
    lw_generator_set_ai_dispatch(&generator, dispatch, &r);
    if (lw_generator_run_scenario(&generator, scenario, NULL, NULL, NULL,
                                  state, fight, outcome) != 0 ||
        r.failed || r.seen != c->count) goto cleanup_state;
    *finished = !r.captured;
    *reward = 0;
    if (*finished) {
        observe(state, observation);
        memset(mask, 0, LEEKRL_LW_ACTION_COUNT);
        if (outcome->winner == 0) *reward = 1;
        if (outcome->winner == 1) *reward = -1;
    }
    rc = 0;
cleanup_state:
    lw_state_dispose(state);
cleanup_scenario:
    lw_scenario_dispose(scenario);
cleanup:
    free(outcome);
    free(fight);
    free(state);
    free(scenario);
    return rc;
}

static int reset_cb(void *context, uint64_t seed, float *o, uint8_t *mask)
{
    leekrl_leekwars *c = context;
    int finished;
    float reward;
    c->seed = (int)(seed & UINT64_C(0x7fffffff));
    c->count = 0;
    return run(c, o, mask, &finished, &reward) || finished;
}

static int step_cb(void *context, size_t action, float *o, uint8_t *mask,
                   float *reward, int *terminated, int *truncated)
{
    leekrl_leekwars *c = context;
    int finished;
    if (c->count >= LW_FIGHT_MAX_TURNS || action >= LEEKRL_LW_ACTION_COUNT)
        return -1;
    c->actions[c->count++] = (unsigned char)action;
    if (run(c, o, mask, &finished, reward)) return -1;
    *terminated = finished;
    *truncated = 0;
    return 0;
}

leekrl_env_status leekrl_leekwars_create(int weapon_id, int max_turns,
                                        leekrl_leekwars **out)
{
    LwWeapon *w;
    leekrl_leekwars *c;
    int i;
    if (!out) return LEEKRL_ENV_INVALID_ARGUMENT;
    *out = NULL;
    w = lw_weapons_get_weapon(weapon_id);
    if (!w || max_turns < 1 || max_turns > LW_FIGHT_MAX_TURNS ||
        w->n_passive_effects || w->base.cost < 0 || w->base.cost > 9 ||
        w->base.attack != &w->attack_storage ||
        w->attack_storage.area_id != LW_AREA_TYPE_SINGLE_CELL ||
        w->attack_storage.max_uses == 0 || w->attack_storage.max_uses < -1 ||
        w->attack_storage.n_effects <= 0 ||
        w->attack_storage.n_effects > LW_ATTACK_MAX_EFFECTS)
        return LEEKRL_ENV_INVALID_ARGUMENT;
    for (i = 0; i < w->attack_storage.n_effects; ++i) {
        int targets = w->attack_storage.effects[i].targets;
        if (w->attack_storage.effects[i].id != LW_EFFECT_TYPE_DAMAGE ||
            !(targets & LW_TARGET_ENEMIES) || !(targets & LW_TARGET_NON_SUMMONS) ||
            (targets & (LW_TARGET_CASTER | LW_TARGET_ALLIES)))
            return LEEKRL_ENV_INVALID_ARGUMENT;
    }
    c = calloc(1, sizeof(*c));
    if (!c) return LEEKRL_ENV_OUT_OF_MEMORY;
    c->weapon_id = weapon_id;
    c->weapon = w;
    c->max_turns = max_turns;
    *out = c;
    return LEEKRL_ENV_OK;
}

void leekrl_leekwars_destroy(leekrl_leekwars *context) { free(context); }

leekrl_env_callbacks leekrl_leekwars_callbacks(void)
{
    leekrl_env_callbacks callbacks = {reset_cb, step_cb};
    return callbacks;
}
