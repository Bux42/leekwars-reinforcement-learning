# Resurrection (postponed)

`CHIP_RESURRECTION` (84) is left out of the first version: it is not in the
action space, and builds cannot equip it.

## Why

Every other castable needs a target cell. Resurrection needs a target cell
**and a dead entity** (`resurrect(entity, cell)` in the game). An action made of
an item and a relative cell cannot say which entity to revive.

The number of dead entities also changes during a fight and can grow large with
summons, while the action space must stay fixed.

It matters little in 1v1, where only summons can be revived (a dead leek takes
its summons with it and the fight ends). It must be solved before team fights
(4v4...), where the chip is very strong.

## What the engine does

- Range 1-2 (12 cells), 18 TP, cooldown 15.
- Any dead entity can be revived, **enemies and their summons included**.
- Max life is halved (more on a critical), and current life is half of that.
- A revived summon counts toward the limit of 8 living summons per team.
- A revived summon plays right after its summoner if the latter is alive.
- C function: `lw_state_resurrect_entity`.

## Open questions

- Dead enemy summons: mask them or not? Reviving one could help with placement,
  for example as an obstacle.
- How to point at the dead entity:
  - **One action per dead entity**, sorted by turn order. Every entity is
    reachable and the agent may learn the link with the entity observation,
    which must then list dead entities in the same order. Needs a maximum number
    of dead entities, and a slot's meaning changes as entities die.
  - **One action per summon type** (12 cells x 13 types = 156 actions), plus one
    per ally leek in team fights. Stable meaning, but cannot tell two dead
    summons of the same type apart. Doubling it would cover enemies.
  - **Two steps**: choose the entity, then the cell. Small action space, but adds
    an intermediate state to the environment.
- What the agent must observe about dead entities.

`CHIP_AWAKENING` (415) works the same way, but only bosses use it.
