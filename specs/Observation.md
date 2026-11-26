# Observation (1v1, first version)

What the environment gives the agent at each decision. Implemented in
[observation.h](../include/leekrl/observation.h).
The reasons for the debatable choices are in [adr/observation.md](../adr/observation.md).

The point of view is always that of the **acting entity**: the leek or summon
whose turn it is. "Ally" and "enemy" are relative to its team.

## Overview

| Block | Type | Size | Content |
|---|---|---|---|
| Map | byte | 2,381 | One category per cell |
| Global | float | 3 | Turn, number of summons |
| Entities | float | 18 x 116 = 2,088 | One block per living entity |
| Items | float | 2 x 284 = 568 | Equipment of each leek |
| Uses left | float | 46 | Limited items of the acting entity |
| **Total** | | **2,705 floats + 2,381 bytes** | |

The legal actions, among 10,566, are provided separately. The network takes them as an
input by default, so the agent sees what it can reach and target. An experiment without
them will tell how much it helps.

What is mostly empty is given as lists: the cells of the map that are not free (about 130)
and the legal actions (about 50). Memory: about 11 KB per decision.

Scaling divides by a fixed number, chosen so usual values fall within about -1 to 1.
Values are not clipped unless stated.

## Turned to face the enemy

By default, everything that has a direction is turned so that the enemy leek is always at
`0 <= dy <= dx`: positions of entities, map, legal actions, and the action chosen. Among
the 8 symmetries of the grid (quarter turns and reflections, see
[symmetry.h](../include/leekrl/symmetry.h)), it is the one that does so for the acting
entity. The rules do not depend on directions, and every action keeps its counterpart:
nothing is lost. See [adr/observation.md](../adr/observation.md).

## Map

All relative positions `(dx, dy)` with `|dx| + |dy| <= 34`, ordered by `dy` then
`dx`, both ascending. 34 is the largest distance between two cells of the map, so the
whole map is always visible. The acting entity is always at index 1190.

| Value | Category |
|---|---|
| 0 | Outside the map |
| 1 | Free |
| 2 | Obstacle |
| 3 | Self |
| 4 | Ally leek |
| 5 | Ally summon, plants included |
| 6 | Enemy leek |
| 7 | Enemy summon, plants included |

"Ally leek" only appears when a summon is acting. The map is translated, then turned.

The network gets the same thing in another form: for each cell that is not free, its index
x 6 + its category - 2, in no particular order; and the position of the map's center,
`(dy + 17) x 35 + dx + 17`, which says which cells are outside.

### Sight (option)

With the `sight` option, also a list of map indices, in no particular order: the cells the
acting entity could move to with its MP, its own included, from which the enemy leek
would be in its line of sight once it has left its own cell. Each gets a vector of its
own, added to the map's. Range, and the rules of items that act on the first entity in a
line (`grapple`, `boxing_glove`), are not taken into account. It follows from the map:
it is given, not guessed, so that the network does not have to learn line of sight
before it can learn where to go. About 4% slower training.

## Global (3 floats)

| Index | Content | Scale |
|---|---|---|
| 0 | Current turn | / 64 |
| 1 | Living ally summons | / 8 |
| 2 | Living enemy summons | / 8 |

## Entities (18 x 116 floats)

18 is the real maximum in 1v1: 2 leeks and at most 8 living summons per team. Unused
blocks are all zeros. Dead entities are not listed.

**Order:** the acting entity first, then the others in the order they will play.
Awakening plants keep their place in that order, although they act differently.

Each block:

| Offset | Floats | Content | Scale |
|---|---|---|---|
| 0 | 1 | Present | 1 |
| 1 | 1 | Ally (1) or enemy (0). The acting entity is an ally | 1 |
| 2 | 14 | Type, one-hot: leek, then summons by id 1 to 13 | 1 |
| 16 | 1 | Life / total life | 1 |
| 17 | 1 | `log(1 + total life)` | / 10 |
| 18 | 1 | `dx` from the acting entity | / 34 |
| 19 | 1 | `dy` from the acting entity | / 34 |
| 20 | 1 | Distance from the acting entity | / 34 |
| 21 | 1 | Available TP. Total TP for awakening plants, which recover it when they awaken | / 30 |
| 22 | 1 | Total TP | / 30 |
| 23 | 1 | Available MP | / 15 |
| 24 | 1 | Total MP | / 15 |
| 25 | 6 | Strength, wisdom, agility, resistance, science, magic | / 1000 |
| 31 | 1 | Frequency | / 1000 |
| 32 | 1 | Power | / 100 |
| 33 | 1 | Damage return | / 100 |
| 34 | 1 | Absolute shield | / 500 |
| 35 | 1 | Relative shield | / 100 |
| 36 | 10 | States (below) | |
| 46 | 62 | Effects (below) | |
| 108 | 1 | Awakening zone, 0 if not an awakening plant | / 10 |
| 109 | 1 | Awakening plant already awakened this turn by the acting entity | 1 |
| 110 | 6 | Summons: cooldown of each of their chips, in the order of the summon data | / 10, at most 1 |

Characteristics are current values, buffs and debuffs included, and can be negative.
Total life decreases with erosion. Cores and RAM are left out: they do nothing once
the fight has started.

The type gives a summon's chips, so only their cooldowns are needed. The zone is 3 for
both awakening plants today, hence 0.3.

### States (5 x 2 floats)

For each state: active (0 or 1), then duration. Only those an entity can have in 1v1:

| Offset | State | Source |
|---|---|---|
| 36 | Unhealable (2) | `hemorrhage` |
| 38 | Dense (6) | `neutronium` |
| 40 | Rooted (9) | Plants, permanent |
| 42 | Sterile (12) | `desert_saber` |
| 44 | Amputated (13) | `glowing_saber` |

### Effects (31 x 2 floats)

For each type, in this order: total amount, then longest duration.

| Types | Amount scale |
|---|---|
| 2 heal over time, 6 absolute shield, 13 poison, 25 aftereffect, 27 absolute vulnerability, 29 stolen shield, 37 raw absolute shield | / 500 |
| 5 relative shield, 20 damage return, 26 vulnerability, 52 raw power, 54 raw relative shield | / 100 |
| 7 MP, 8 TP, 17 MP shackle, 18 TP shackle, 31 raw MP, 32 raw TP | / 10 |
| 3, 4, 19, 21, 22, 38 to 42, 44, 47, 48: buffs and shackles of characteristics | / 500 |

The blocks themselves are ordered by type id:
2 3 4 5 6 7 8 13 17 18 19 20 21 22 25 26 27 29 31 32 37 38 39 40 41 42 44 47 48 52 54.

Amounts are positive, even for shackles: the type says what they do. Scaled amounts
are limited to 5, so a very large stack does not swamp the network.

### Durations

Turns / 10, at most 0.9. Permanent is 1.0, none is 0.0. The longest duration of an
item today is 7 turns.

A duration does not always mean "turns of the entity carrying it": poison, aftereffect
and heal over time count the turns of the target, all others the turns of the caster.

## Items (2 x 284 floats)

First the leek of the acting entity's team, then the enemy leek. They keep these
places when a summon is acting.

| Offset | Floats | Content |
|---|---|---|
| 0 | 142 | Equipped (0 or 1): 37 weapons then 105 chips, sorted by item id |
| 142 | 105 | Chip cooldown: turns left / 10, at most 1. Same order |
| 247 | 37 | Weapon in hand, one-hot. All zeros if none |

Items are those of the 1v1 action space. A chip usable once per fight stays at 1.0
afterwards. Cooldowns shared by a team are included.

The enemy's equipment and cooldowns are visible, as in the real game.

## Uses left (46 floats)

For each item with a limited number of uses per turn: uses left for the acting entity,
/ 5. All 37 weapons, then 9 chips, both sorted by item id. 0 if the entity does not
have the item. Other entities have no such floats: their uses reset before they play
again.

## End of the fight

The observation is all zeros once the fight is over. It must not be used to estimate
a value: a finished fight has none.

## Safety checks

The environment must fail with an error, never silently drop information, if:

- more than 18 entities are alive,
- a living entity has an effect type or a state that is not listed above,
- a leek carries an item that is not in the action space,
- any float is not finite.

Adding items to the game can trigger the second one: the layout then has to change,
and trained models no longer fit.

## Not included

- **Line of sight and walking distance** per cell. The mask shows the cells that can be
  reached or targeted now, but not for actions that are too expensive or on cooldown,
  and never from the enemy's side. The sight option only says, for reachable cells,
  whether the enemy leek can be seen.
- **Mirroring.** The same situation facing the other way looks different.
- **Number of actions already played** this turn.
- **Who cast an effect**, and everything listed in the ADR.
- **Summoner of a summon.** Obvious in 1v1, needed in team fights.
- **Dead entities.** Needed with [resurrection](postponed/resurrection.md).

## Checks

In [test_observation.c](../tests/test_observation.c):

1. Offsets and sizes match this document.
2. Translation: moving a whole situation elsewhere on the map changes only the cells
   outside the map.
3. Symmetry: the same situation seen by each team swaps ally and enemy.
4. Random fights: every value compared with an independent reading of the engine,
   turn order included.
5. Effects and states, permanent ones included.
6. Situations that must be refused.
7. Same seed and actions give byte-identical observations.
