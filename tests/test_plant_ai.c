#include "leekrl/actions.h"
#include "leekrl/item_snapshot.h"
#include "leekrl/plant_ai.h"
#include "lw_bulb_template.h"
#include "lw_chip.h"
#include "lw_effect.h"
#include "lw_fight.h"
#include "lw_generator.h"
#include "lw_map.h"
#include "lw_outcome.h"
#include "lw_scenario.h"
#include "lw_state.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

enum { CORN = 164, CHILLI = 165, PIQUANT = 444, CAPSAICIN = 445,
       SUGAR = 446, POPCORN = 447 };

typedef struct {
    LwGenerator generator;
    LwScenario scenario;
    LwState state;
    LwFight fight;
    LwOutcome outcome;
    LwEntity *caster;
    LwEntity *plant;
    int casts[8];
    int n_casts;
} fixture;

/* Three leeks per team, parked far away until a test places them. */
static fixture *make_fixture(void)
{
    fixture *f = calloc(1, sizeof(*f));
    LwEntityInfo info;
    int team, i;
    CHECK(f);
    lw_generator_init(&f->generator);
    lw_generator_set_plant_ai_dispatch(&f->generator, leekrl_plant_ai, NULL);
    lw_scenario_init(&f->scenario);
    f->scenario.seed = 5;
    f->scenario.type = LW_FIGHT_TYPE_TEAM;
    f->scenario.map.present = 1;
    for (team = 0; team < 2; ++team) {
        for (i = 0; i < 3; ++i) {
            lw_entity_info_init(&info);
            info.id = 1 + team * 3 + i;
            info.life = 2000;
            info.tp = 20;
            info.mp = 10;
            info.ram = 10;
            info.level = 301;
            if (info.id == 1) {
                info.chips[0] = CORN;
                info.chips[1] = CHILLI;
                info.n_chips = 2;
            }
            CHECK(lw_scenario_add_entity(&f->scenario, team, &info));
        }
    }
    CHECK(!lw_generator_prepare_scenario(&f->generator, &f->scenario, NULL, NULL, NULL,
                                         &f->state, &f->fight, &f->outcome));
    CHECK(!lw_fight_step_start(&f->fight));
    for (;;) {
        CHECK(!lw_fight_step_begin_turn(&f->fight, &f->caster));
        CHECK(f->caster);
        if (f->caster->m_id == 1) break;
        CHECK(!lw_fight_step_end_turn(&f->fight));
    }
    /* The map has no obstacle. Keep entities out of the way on the border. */
    for (i = 0; i < f->state.n_entities; ++i)
        lw_map_remove_entity(f->state.map, f->state.m_entities[i]);
    for (i = 0; i < f->state.n_entities; ++i)
        lw_map_set_entity(f->state.map, f->state.m_entities[i],
                          lw_map_get_cell(f->state.map, i));
    return f;
}

static void free_fixture(fixture *f)
{
    lw_state_dispose(&f->state);
    lw_scenario_dispose(&f->scenario);
    free(f);
}

static LwEntity *by_id(fixture *f, int id)
{
    int i;
    for (i = 0; i < f->state.n_entities; ++i)
        if (f->state.m_entities[i]->m_id == id && !lw_entity_is_summon(f->state.m_entities[i]))
            return f->state.m_entities[i];
    CHECK(0);
    return NULL;
}

static LwCell *cell_at(fixture *f, int dx, int dy)
{
    LwCell *center = lw_map_get_cell(f->state.map, 306);
    LwCell *cell = lw_cell_next(center, f->state.map, dx, dy);
    CHECK(cell);
    return cell;
}

static void place(fixture *f, int id, int dx, int dy)
{
    LwEntity *e = by_id(f, id);
    lw_map_remove_entity(f->state.map, e);
    lw_map_set_entity(f->state.map, e, cell_at(f, dx, dy));
}

/* Plant at the center, cast from (-4, 0), outside the zone. Records the chips
 * the plant used, in order, from the action log. */
static void plant(fixture *f, int chip_id)
{
    leekrl_action action = {LEEKRL_ACTION_CHIP, 0, 4, 0};
    LwActionStream *log = &f->state.actions.stream;
    int i, result, first = log->n_entries;
    LwChip *chip;
    action.item_id = chip_id;
    place(f, 1, -4, 0);
    CHECK(leekrl_action_execute(&f->fight, f->caster, &action, &result) == LEEKRL_ACTION_OK);
    CHECK(result > 0 && !f->state.fatal_error);
    f->plant = lw_state_get_last_entity(&f->state);
    CHECK(f->plant->awakening_zone == 3 && f->plant->has_summon_ai);
    f->n_casts = 0;
    for (i = first; i < log->n_entries; ++i) {
        int k;
        if (log->entries[i].type != LW_ACTION_USE_CHIP) continue;
        CHECK(f->n_casts < 8);
        for (k = PIQUANT; k <= POPCORN; ++k) {
            chip = lw_chips_get_chip(k);
            if (log->entries[i].v[0] == lw_chip_get_template(chip))
                f->casts[f->n_casts++] = k;
        }
    }
}

static void hurt(fixture *f, int id, int missing) { by_id(f, id)->life -= missing; }

static void test_chilli(void)
{
    fixture *f = make_fixture();
    place(f, 4, 1, 0);                              /* adjacent: area wins */
    plant(f, CHILLI);
    CHECK(f->n_casts == 1 && f->casts[0] == CAPSAICIN);
    CHECK(by_id(f, 4)->life < 2000);
    free_fixture(f);

    f = make_fixture();
    place(f, 4, 3, 0);                              /* far: two piquant win */
    plant(f, CHILLI);
    CHECK(f->n_casts == 2 && f->casts[0] == PIQUANT && f->casts[1] == PIQUANT);
    free_fixture(f);

    f = make_fixture();
    place(f, 4, 3, 0);                              /* two far: area wins */
    place(f, 5, 0, 3);
    plant(f, CHILLI);
    /* Two entities in the zone: a second awakening, area chip on cooldown. */
    CHECK(f->n_casts == 3 && f->casts[0] == CAPSAICIN);
    CHECK(f->casts[1] == PIQUANT && f->casts[2] == PIQUANT);
    CHECK(by_id(f, 4)->life < 2000 && by_id(f, 5)->life < 2000);
    free_fixture(f);

    f = make_fixture();
    place(f, 4, 1, 0);                              /* kill first */
    place(f, 5, 0, 1);
    hurt(f, 5, 1990);
    plant(f, CHILLI);
    CHECK(f->n_casts == 2 && f->casts[0] == PIQUANT && f->casts[1] == PIQUANT);
    CHECK(by_id(f, 5)->life <= 0 && by_id(f, 4)->life < 2000);
    free_fixture(f);

    f = make_fixture();
    place(f, 2, 1, 0);                              /* allies only: nothing */
    hurt(f, 2, 500);
    plant(f, CHILLI);
    CHECK(f->n_casts == 0 && by_id(f, 2)->life == 1500);
    free_fixture(f);

    f = make_fixture();
    place(f, 4, 1, 0);                              /* invincible: nothing */
    by_id(f, 4)->states |= 1u << LW_ENTITY_STATE_INVINCIBLE;
    plant(f, CHILLI);
    CHECK(f->n_casts == 0 && by_id(f, 4)->life == 2000);
    free_fixture(f);
}

static void test_corn(void)
{
    fixture *f = make_fixture();
    place(f, 2, 3, 0);                              /* one far ally: sugar twice */
    hurt(f, 2, 1500);
    plant(f, CORN);
    CHECK(f->n_casts == 2 && f->casts[0] == SUGAR && f->casts[1] == SUGAR);
    CHECK(by_id(f, 2)->life > 500);
    free_fixture(f);

    f = make_fixture();
    place(f, 2, 1, 0);                              /* two allies: popcorn */
    place(f, 3, 0, 1);
    hurt(f, 2, 1500);
    hurt(f, 3, 1500);
    plant(f, CORN);
    CHECK(f->n_casts == 3 && f->casts[0] == POPCORN);
    CHECK(f->casts[1] == SUGAR && f->casts[2] == SUGAR);
    CHECK(by_id(f, 2)->life > 500 && by_id(f, 3)->life > 500);
    free_fixture(f);

    f = make_fixture();
    place(f, 2, 1, 0);                              /* nobody injured, hurt enemy */
    place(f, 4, 0, 1);
    hurt(f, 4, 1500);
    plant(f, CORN);
    CHECK(f->n_casts == 0 && by_id(f, 4)->life == 500);
    free_fixture(f);

    f = make_fixture();
    place(f, 2, 1, 0);                              /* unhealable: nothing */
    hurt(f, 2, 1500);
    by_id(f, 2)->states |= 1u << LW_ENTITY_STATE_UNHEALABLE;
    plant(f, CORN);
    CHECK(f->n_casts == 0 && by_id(f, 2)->life == 500);
    free_fixture(f);
}

/* Entering the zone awakens the plant in the middle of the movement. */
static void test_enter_zone(void)
{
    fixture *f = make_fixture();
    leekrl_action move = {LEEKRL_ACTION_MOVE, 0, 2, 0};
    int result;
    hurt(f, 1, 1000);
    plant(f, CORN);
    CHECK(f->n_casts == 0 && f->caster->life == 1000);
    CHECK(leekrl_action_execute(&f->fight, f->caster, &move, &result) == LEEKRL_ACTION_OK);
    CHECK(result == 2 && f->caster->life > 1000 && !f->state.awakening_plant);
    CHECK(lw_entity_get_cell(f->caster) == cell_at(f, -2, 0));
    free_fixture(f);
}

int main(void)
{
    char error[512];
    if (!leekrl_item_snapshot_register(error, sizeof(error))) {
        fprintf(stderr, "item registration: %s\n", error);
        return 1;
    }
    test_chilli();
    test_corn();
    test_enter_zone();
    leekrl_item_snapshot_clear();
    return 0;
}
