## First implementation

The first implementation will be quite generic.
The game has "castables", weapons and chips, that can be casted on a cell / entity and affect the game (boost, damages etc)

There are currently 36 weapons in the game and 106 chips.
The creator of the game can change the game at any time, change the rules, mechanics, add a weapon, or a chip etc.

That is around 142 different castables.

In leekwars, an entity (leek) can have up to 4 equipped weapons and up to 32 equipped chips (actually, the theorical limit for total possible equipped chips is around 50, but we'll ignore that for now, it might be interesting to revisit the generator hardcoded 32 equipped chips limit in the future)

Since we want to train an agent to be "the best" in any kind of situation, during training, the leek weapons, chips, stats etc will be "randomized".

A unique model will be trained for each fight type (1v1, 4v4 etc), starting with 1v1 fights (simplest fight type).
We do not care about generalizing accross multiple different fight types.

The hope is that some build + strategy will emerge, some new interesting strategies will be found, etc.

Since we want to generalize accross (for example) 1v1 fights, the proposition is to start by including ALL castables in the agent's action space.

Here is what the action space would look like:
Action 0: end turn
Action 1 - 841: movement actions (840 accessible cells assuming the max movement points the agent will ever have in fights is 20)
Action 841 - 954: weapon pistol actions (113 reachable cells with range 1 - 7)
etc until all castables that the agent might have is in the action space

So every possible action must be present.

Then, using action masking, we will trim it down, mask off castables that aren't equipped, or aren't usable at a given moment
Also we will mask out "illegal" actions (moving out of the map boundaries, casting on an obstable tile etc) so that the agent can only pick "relevant" actions.
Furthermore, we can also trim down "100% useless" actions, with some caveat.
For example, attacking an empty cell is legal, but useless in 99% of the cases, it doesn't affect the fight in any way.
Some exceptions, for example, WEAPON_SWORD that gives 8% RAW_RELATIVE_SHIELD to the caster for 2 turns regarding of entity reached by the cast, you benefit from the shield effect by casting in the void.
There are other castables that enters this category.

Both the actions and the observation will be "relative" to the agent's current position in game, meaning it will be from it's point of vue.

What that means is that (for example) if action 1 is "move on the tile up", when the agent picks it, the result will always be it moving one tile up. Same for casting, the actions will represent casting positions RELATIVE to the agent's POV.

The reason is better generalization during training (and efficiency).

The observation will also be relative, the agent must see the whole map, from it's perspective, centered on it's position, similar to how the action space works.

The map observation will contain floats representing what is on which tile, with few variations, for example, we want (for a start) the following categories:

- Free cell
- Obstacle cell
- Ally entity (excluding self)
- Ally summon entity
- Enemy entity
- Enemy summon entity

The observation will also contain all relevant information about all entities, including itself.

The entity obs will start with the current entity, then all other entities.
The other entities will be sorted by order of turn, such as the next playing entities will be "first" in the entity observation order.

Since some chips can be used to summon other entities, controlled by the caster, we're gonna need to precompute / guesstimate the max possible number of entities in a fight, to reduce the chance of the entity obs "overflowing" in a fight where there are 50 entities.
I think that, same as the arbitrary 20 movement points and 32 equipped chips limit, we should find a reasonable number, I don't think it's relevant to be able to handle 100 entities in a fight when there is 0.0001% chance of it happening.

It will contain all relevant information:

- Lifepoints percentage
- Equipped weapon
- Vertical & horizontal position relative to agent
- Total life log float (not sure about this one, but the agent should differenciate between an entity that has 100/100 lifepoints and another that has 10 000 / 10 000 lifepoints)
- Available TP percentage (assuming an "upper bound" to compute percentage from?)
- Available MP percentage (assuming an "upper bound" to compute percentage from?)
- All the other stats, strenght, agility etc
- 4 floats representing equipped weapons
- 64 floats representing equipped chips, one float unique representation of a chip + it's current cooldown
- two floats per possible state, 1.0 = active, 0.0 = inactive + their duration
- two floats per possible active effects, one for duration, one representing the amount (not all effects scale the same way, some are stackable, etc)

Additionnaly, the current turn number might be relevant for the agent to see (a fight has a limit of 64 max turns, ending it in a draw when reached).

And in the past, I've noticed that putting the action mask itself in the agent's observation space helped a lot, so we might want to check that theory again.

## Training randomization for generalization and emergence

For the actual fights during training, the randomization of those fights will be important.

To start, we're gonna want each leek to have 4 unique weapons equipped, with a random pick amongst all available weapons (there is a rule that forbids having multiple legendary weapons equipped that needs to be respected).

For the chips, as the number of chips one entity can equip is directly linked to the entity's RAM, we have to first generate the entity's stats.

Important: Some chips are only relevant in specific fight types. For example, CHIP_EXASPERATION can only be equipped on a specific type of boss. It would be irrelevant to include it in 1v1 fights. The WEAPON_SCYTHE is also an exception, this weapon can only be used by the pumpkin boss.

We're gonna only train entities at level 301 to start with, meaning there will be 1780 available capital points, that can be spent in the following characteristics: life, strength, wisdom, agility, resistance, science, magic, frequency, cores, ram, tp, mp

Not each characteristic attribution costs the same, and there are threshold that, when reached, increase the cost for the subsequent point allocations.

The entity stats will be attributed such there aren't enough available capital points to raise any individual stat.

The game has "components", which are equipable items that gives additionnal stats points.
You can only equip up to 8 unique components at a time.
We will exclude legacy COMPONENT_RECOVERY_CORE and COMPONENT_RECOVERY_RAM.
We will pick 8 random components.
We might also want to exclude "irrelevant" components like the COMPONENT_RGB that only gives one additionnal health point.

Components can be altered, meaning it's stats can be changed.
To start, we will disregard this mechanic (but in the future, we will for sure play around with it), as I don't really understand it, but it is possible to (for example) add an additionnal TP to a component.

To keep the "max 32 chips" arbitrary rule respected, that randomized stats attribution shouldn't exceed 32 RAM.
Or the equipped chips randomization musn't exceed 32 chips. No duplicate equipped chips are allowed.

I also want the training to randomly pick a full build config amongst a list of "known" builds, provided by me.
There are some specific builds that are the "meta" in the game (poison builds, OTK builds, nova builds etc). And having the agent pick those builds sometimes (don't know what the "good" probability of picking those build would be) will be very interesting, as we will see how the agent plays on those known builds.

The generation (obstacles and fight placements) must also be random (except for some exceptions, like for example, the boss fights, but that's for waaaay later)

## Reward shaping

The first version of the reward will be the following:
Winning a fight gives big reward, losing a fight loses big reward (main goal is to win)
When winning a fight, additionnal reward will be given based on:

- Fight duration (shorted fights -> bigger reward, encouraging more efficient actions, thus indirectly speeding up training)
- Total actions (if an agent can move to it's destination in 1 action, but does it in 3 consecutive movements, we want it to favor one action instead)

Killing an entity should give fixed reward (smaller for summons)
Getting an ally killed should gived fixed negative reward (smaller for summons as well)

Inflicting damage gives reward proportionnal to the damage
Recieving damage loses reward proportionnal to the damage

There are different types of damage (direct, poison, after effect, nova etc)
Not sure how each should be dealt with in terms of reward, but it should "steer" the agent and help it converge faster, as fights can be quite long.
