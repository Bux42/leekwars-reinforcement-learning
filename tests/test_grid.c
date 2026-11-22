#include "leekrl/grid.h"

#include "lw_entity.h"
#include "lw_fight.h"
#include "lw_map.h"
#include "lw_state.h"
#include "lw_team.h"

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

typedef struct {
    LwState state;
    LwMap map;
    LwFight fight;
    LwEntity actor;
    LwEntity ally;
    LwEntity enemy;
    LwEntity dead;
    LwTeam teams[2];
} fixture;

static fixture *make_fixture(void)
{
    fixture *f = calloc(1, sizeof(*f));
    CHECK(f);
    lw_state_init(&f->state);
    lw_map_init(&f->map, 18, 18);
    f->map.state = &f->state;
    f->state.map = &f->map;
    lw_fight_init(&f->fight, NULL);
    lw_fight_set_state(&f->fight, &f->state);

    lw_entity_init_id_name(&f->actor, 1, "actor");
    lw_entity_init_id_name(&f->ally, 2, "ally");
    lw_entity_init_id_name(&f->enemy, 3, "enemy");
    lw_entity_init_id_name(&f->dead, 4, "dead");
    lw_entity_set_state(&f->actor, &f->state, 0);
    lw_entity_set_state(&f->ally, &f->state, 1);
    lw_entity_set_state(&f->enemy, &f->state, 2);
    lw_entity_set_state(&f->dead, &f->state, 3);
    lw_entity_set_team(&f->actor, 0);
    lw_entity_set_team(&f->ally, 0);
    lw_entity_set_team(&f->enemy, 1);
    lw_entity_set_team(&f->dead, 1);
    lw_team_init(&f->teams[0]);
    lw_team_init(&f->teams[1]);
    f->state.teams[0] = &f->teams[0];
    f->state.teams[1] = &f->teams[1];
    f->state.n_teams = 2;
    lw_team_add_entity(&f->teams[0], &f->actor);
    lw_team_add_entity(&f->teams[0], &f->ally);
    lw_team_add_entity(&f->teams[1], &f->enemy);
    lw_team_add_entity(&f->teams[1], &f->dead);
    f->actor.life = f->actor.m_total_life = 100;
    f->ally.life = f->ally.m_total_life = 100;
    f->enemy.life = f->enemy.m_total_life = 100;
    f->dead.life = 0;
    f->dead.m_total_life = 100;
    f->state.m_entities[f->state.n_entities++] = &f->actor;
    f->state.m_entities[f->state.n_entities++] = &f->ally;
    f->state.m_entities[f->state.n_entities++] = &f->enemy;
    f->state.m_entities[f->state.n_entities++] = &f->dead;

    {
        LwCell *origin = lw_map_get_cell(&f->map, 306);
        lw_map_set_entity(&f->map, &f->actor, origin);
        lw_map_set_entity(&f->map, &f->ally,
                           lw_map_get_next_cell(&f->map, origin, -1, 0));
        lw_map_set_entity(&f->map, &f->enemy,
                           lw_map_get_next_cell(&f->map, origin, 2, 0));
        lw_entity_set_cell(&f->dead,
                           lw_map_get_next_cell(&f->map, origin, 0, 1));
    }
    return f;
}

static size_t index_at(int radius, int dx, int dy)
{
    int side = 2 * radius + 1;
    return (size_t)((dy + radius) * side + (dx + radius));
}

int main(void)
{
    fixture *f = make_fixture();
    leekrl_grid_layout layout;
    float *grid;
    size_t required, i;
    CHECK(leekrl_grid_layout_for(-1, &layout) == LEEKRL_GRID_INVALID_ARGUMENT);
    CHECK(leekrl_grid_layout_for(33, &layout) == LEEKRL_GRID_INVALID_ARGUMENT);
    CHECK(leekrl_grid_layout_for(2, &layout) == LEEKRL_GRID_OK);
    CHECK(layout.radius == 2 && layout.height == 5 &&
           layout.width == 5 && layout.channels == LEEKRL_GRID_CHANNEL_COUNT);
    CHECK(leekrl_grid_observation(&f->fight, &f->actor, 2, NULL, 0, &required) ==
           LEEKRL_GRID_OK);
    CHECK(required == 150);
    grid = calloc(required, sizeof(*grid));
    CHECK(grid);

    CHECK(leekrl_grid_observation(&f->fight, &f->actor, 2, grid, required - 1,
                                   &required) == LEEKRL_GRID_BUFFER_TOO_SMALL);
    CHECK(leekrl_grid_observation(&f->fight, &f->actor, 2, grid, 150, &required) ==
           LEEKRL_GRID_OK);
    CHECK(required == 150);
    for (i = 0; i < required; ++i)
        CHECK(grid[i] == 0.0f || grid[i] == 1.0f);

    {
        int base = lw_cell_get_x(lw_entity_get_cell(&f->actor));
        int oy = lw_cell_get_y(lw_entity_get_cell(&f->actor));
        LwCell *ally = lw_entity_get_cell(&f->ally);
        LwCell *enemy = lw_entity_get_cell(&f->enemy);
        LwCell *dead = lw_entity_get_cell(&f->dead);
        size_t self_index = index_at(2, 0, 0);
        size_t ally_index = index_at(2, lw_cell_get_x(ally) - base,
                                     lw_cell_get_y(ally) - oy);
        size_t enemy_index = index_at(2, lw_cell_get_x(enemy) - base,
                                      lw_cell_get_y(enemy) - oy);
        size_t dead_index = index_at(2, lw_cell_get_x(dead) - base,
                                     lw_cell_get_y(dead) - oy);
        CHECK(grid[LEEKRL_GRID_SELF * 25 + self_index] == 1.0f);
        CHECK(grid[LEEKRL_GRID_ALLY * 25 + ally_index] == 1.0f);
        CHECK(grid[LEEKRL_GRID_ENEMY * 25 + enemy_index] == 1.0f);
        CHECK(grid[LEEKRL_GRID_DEAD * 25 + dead_index] == 1.0f);
    }

    {
        LwCell *origin = lw_entity_get_cell(&f->actor);
        LwCell *obstacle = lw_map_get_next_cell(&f->map, origin, 1, 0);
        CHECK(obstacle);
        lw_cell_set_obstacle(obstacle, 1, 1);
        CHECK(leekrl_grid_extract(&f->fight, &f->actor, 2, grid, 150, &required) ==
               LEEKRL_GRID_OK);
        CHECK(grid[LEEKRL_GRID_OBSTACLE * 25 + index_at(2, 1, 0)] == 1.0f);
    }

    {
        float edge_grid[54];
        size_t boundary_count = 0;
        lw_map_set_entity(&f->map, &f->actor, lw_map_get_cell(&f->map, 0));
        CHECK(leekrl_grid_extract(&f->fight, &f->actor, 1, edge_grid, 54,
                                   &required) == LEEKRL_GRID_OK);
        for (i = 0; i < 9; ++i)
            boundary_count += edge_grid[LEEKRL_GRID_BOUNDARY * 9 + i] != 0.0f;
        CHECK(boundary_count > 0);
    }

    CHECK(leekrl_grid_status_string(LEEKRL_GRID_OK));
    free(grid);
    lw_state_dispose(&f->state);
    lw_entity_dispose(&f->actor);
    lw_entity_dispose(&f->ally);
    lw_entity_dispose(&f->enemy);
    lw_entity_dispose(&f->dead);
    free(f);
    return 0;
}
