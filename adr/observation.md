# Observation decisions

Choices that simplify what the agent sees and may influence training. To be
revisited if the agent misplays in ways they could explain.

## Active effects: total amount and longest duration per type

Date: 2026-11-24. Status: accepted for the first version.

### Context

An entity can carry several active effects of the same type, with different
amounts and remaining durations: for example two poisons from two casts. Their
number varies during a fight (up to 64 per entity in the generator), while the
observation has a fixed size.

### Decision

For each entity and each effect type, the observation holds two floats:

- the **sum** of the amounts of its active effects of that type,
- the **longest** remaining duration among them.

The same floats exist for every entity, the one currently playing included.

Some effects are permanent (duration -1 in the generator). They count as longer
than any other duration, so they must not be compared as plain numbers.

Only types that can stay active on an entity are listed. There are 31, plus
states, which have floats of their own: 2 3 4 5 6 7 8 13 17 18 19 20 21 22 25 26
27 29 31 32 37 38 39 40 41 42 44 47 48 52 54. All were seen active on the entity
currently playing in 3,000 random fights, and no other type was seen on any
living entity. Instant effects (damage, teleport, push...) leave nothing active.
Passive effects of weapons (33 to 36, 50, 55, 56, 58, 63) never become active
themselves: they create an effect of another type, which is visible. This list
must be checked again when items change.

### What the agent cannot see

- **How the amount is spread over time.** "100 poison for 1 turn plus 20 for 4
  turns" and "20 poison for 1 turn plus 100 for 4 turns" look the same: 120 for
  4 turns. Future poison damage, or when a shield drops, can be misjudged.
- **How many effects make up the total**, which matters for what reduces,
  removes or steals effects one by one.
- **Who cast them.** Effects disappear when their caster dies, so killing a
  summon may remove part of the total.
- **Which of them cannot be removed** (irreducible effects).

- **Whose turns a duration counts.** Poison, aftereffect and heal over time count the
  turns of the target, all others those of the caster.

### Scenario to test

Two enemies can both cast the same poison. A has 1 life and cast 10,000 poison on the
agent. B has 10,000 life and cast none. Killing A removes the poison and saves the
agent. **The layout does not contain this information**, so the agent cannot learn it
in general: more training will not help.

In 1v1 the case is limited, since killing the enemy leek ends the fight. It remains for
the summons that poison: wizard bulb (`venom`, `toxin`) and chilli pepper
(`capsaicin`). It matters fully in team fights. The same goes for buffs and shields
received from an ally about to die.

Candidate fix: two floats per entity, the poison and aftereffect it cast on the acting
entity (total amount, longest duration), 36 floats in all.

### Alternatives

- Sum of amount x remaining duration as a third float (total future value).
- Amounts per remaining duration: 1 turn, 2 turns, 3 turns, 4 and more.
- A fixed number of slots per entity, one per effect. Complete until it
  overflows, but larger, and the meaning of a slot changes.

### Signs it should be revisited

- Wasted healing or antidotes against poison, or deaths to foreseeable poison.
- Attacks made just before a shield expires instead of just after.
- Poor timing of buffs, such as renewing a buff that is mostly still active.

## Map: whole-map diamond, one category byte per cell

Date: 2026-11-24. Status: first choice to try, others to be compared.

### Context

The agent must see the whole map from its own position. A previous project used a
flat 1D observation and no convolution, to keep the network small and fast.

### Decision

- 2,381 relative cells (a diamond of radius 34). The 69x69 square holds the same
  information; its other half is always outside the map.
- The environment writes one category per cell, as a byte.
- The network turns each category into 4 learned numbers, then uses dense layers.

A single float per cell would place unrelated categories on one scale. Bytes keep the
memory small: 2.4 KB per observation instead of 76 KB for one-hot floats.

### Measured

Forward pass of a small policy network on CPU, batches of 256. Speed only: it says
nothing about playing strength, and a GPU will change the ranking.

| Map input | Map weights | Time per observation |
|---|---|---|
| 69x69, one float per cell | 1.2M | 38 us |
| Radius 20 (841 cells), one-hot | 1.7M | 46 us |
| 69x69, 4 learned numbers per cell | 4.9M | 59 us |
| 69x69, one-hot | 9.8M | 153 us |
| 69x69, small CNN | 0.7M | 165 us |

The chosen diamond has half the cells of the third row. It was only timed inside a
full network (6.2M weights): about 60 us per observation.

### Weak points

- Dense layers must learn the same local pattern again at each position: cover next to
  the enemy and cover next to the agent share nothing. A CNN would share it.
- Line of sight and walking distances are not given. The action mask, which the network
  takes as an input, shows the cells the agent can reach or target now. The rest must be
  inferred: for actions not currently possible, and from the enemy's side.

### Alternatives

A CNN on the 69x69 square, the radius 20 map (as far as one move action or any cast of
the first version reaches), or extra inputs per cell. Only the last one changes what
the environment writes.

## Items: one fixed position per item

Date: 2026-11-24. Status: first choice to try, others to be compared.

### Context

The first idea was one float per equipment slot, holding a number that identifies the
item. The network would have to recognize precise values in any slot, and close values
would mean unrelated items, whatever the sorting (by level: `whip` next to `grapple`).

### Decision

Each weapon and chip of the 1v1 action space has its own position: 1 if equipped. Chip
cooldowns have their own positions too. The first dense layer then learns what each
item means, and reading the opponent's build is direct.

For a summon, its type and the cooldowns of its chips are enough.

### Cost

568 floats for both leeks, against 136 for the first idea. Adding an item to the game
changes the size, but it changes the action space anyway.

### Alternatives

Item indices per slot with learned vectors, which keep the same size when items are
added and could include item statistics (range, cost...) to help with new items.

## Action mask as a network input

Date: 2026-11-24. Status: accepted, to be confirmed by an experiment without it.

The mask already forbids illegal actions. Giving it to the network as well shows the
agent what it can reach and target, which helped a lot in a previous project.

Cost measured on CPU with the same network: 2.8M more weights (6.2M to 9.0M), and
roughly 10 to 30% more time to choose an action, 10% to learn. Timings vary by about as
much between runs.

## Uses left of limited items

Date: 2026-11-24. Status: accepted.

46 items can be used a limited number of times per turn. The mask only says whether one
more use is possible, so 1 and 4 shots left look the same. The observation gives the
uses left, for the acting entity only: those of others reset before they play again.
