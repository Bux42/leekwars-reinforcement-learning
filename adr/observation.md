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
- The network sums one learned vector per (cell, category), for obstacles and entities
  only, plus one for the acting entity's position on the map.

The map's shape is fixed, so the position says exactly which cells are outside, and free
cells need nothing. Nothing is lost, which a test checks.

The first choice was 4 learned numbers per category followed by dense layers. The new one
costs several times less to learn from on CPU, and can represent more.

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

## Useless casts are masked

Date: 2026-11-24. Status: accepted.

A cast whose area holds no entity any of its effects accepts cannot change anything. It is
masked, except when it affects its caster anyway (a sword's shield), summons or teleports,
or when the caster holds a weapon that reacts to its critical hits. 518,000 masked casts
were executed on cloned fights, none had an effect. 84% of legal casts are masked.

What `j_laser` and `quantum_rifle` steal for their caster depends on who they reach: they
are masked too when they reach nobody.

Casts that reach someone but achieve nothing (healing a leek at full life) are kept.

## Harmful casts are masked

Date: 2026-11-24. Status: accepted, after a review item by item. `--trim useless` goes
without.

Casts that can only harm their own team, or only help the enemy, are masked. One that has
anything else to offer stays, like a `tranquilizer` that reaches an ally and an enemy.

| Masked | Items |
|---|---|
| It only reaches its team, and only takes TP, MP or characteristics from it, or makes it unhealable | `tranquilizer`, `soporific`, `slow_down`, `ball_and_chain`, `fracture`, `crushing`, `brainwashing`, `hemorrhage`, `plutonium_bazooka` |
| It only reaches enemies, and only gives them life, maximum life, TP or characteristics, or removes their shackles | `bandage`, `cure`, `drip`, `therapy`, `regeneration`, `explorer_rifle`, `armoring`, `elevation`, `mutation`, `transmutation`, `knowledge`, `prism`, `wizardry`, `manumission` |
| It only harms its caster, a leek | `shock`, `pebble`, `ice`, `spark`, `flash`, `devil_strike`, `plasma`, `toxin`, `plague`, `gazor`, `desintegration` on itself; `punishment`, `dark_katana` on an empty cell |

Such a cast can still be a strategy. It is kept in these cases:

| Kept | Why |
|---|---|
| The caster has `unstable_destroyer` | Its critical hits heal it, whatever it casts |
| The target already has the effect of the same item, and it does not stack: `tranquilizer`, `hemorrhage`, `knowledge`, `prism`, `wizardry` | The new effect replaces it, even that of the other team: a weak `tranquilizer` on oneself removes a strong one |
| Damage to a leek with `revoked_m_laser`, `plutonium_bazooka`, `mysterious_electrisor` or `unbridled_gazor` | It gives strength, resistance, a shield or magic |
| Poison on a leek with `illicit_grenade_launcher` or `unbridled_gazor` | It gives science or magic |
| Maximum life taken from a leek with `unbridled_gazor` | It gives magic |
| `covid` on oneself | It propagates, and keeps a stronger one away |
| A summon that harms itself | Its death frees a place |
| Life or maximum life given to an enemy while a living leek has `enhanced_lightninger` | It changes which poison kills, hence who gets the TP |
| Magic given to an enemy while someone carries `covid` | Its own `covid` can come back to it, with its magic of the moment |

Having the weapon is enough, in hand or not. Kept on purpose: harming one's own summons (life
steal, places, the TP of `enhanced_lightninger`), `bramble` on an enemy, `superinfection`,
`liberation`, `neutronium`, `repotting`, vulnerability, and whatever moves someone.

Checked by `tests/test_useful.c`, where leeks hold these items in two fights out of three. The
list is written there a second time, item by item: every cast of every decision must be
masked or kept as it says, so an item added to the game and masked by these rules fails the
test until it is looked at. Masked casts are also followed on two clones of the fight, with and
without the cast, where everyone then only ends their turns: nothing must ever be better for
the caster's team, nor worse for its enemies. 10.6 million masked casts were followed, for 32
turns on average, over 60,000 fights: none helped. The last two exceptions were found this way.

This cannot see what only shows when someone acts, for example keeping a harmless enemy
summon alive so that it goes on holding a place, or blocking a corridor.

Results against a random opponent cannot be compared with those of earlier runs: it no longer
kills itself as much.

## One independent output per action

Date: 2026-11-24. Status: accepted as the baseline.

Each of the 10,566 actions has its own output, as planned from the start: every castable on
every cell can have its own consequences. Other forms will be compared to it later.

The alternative, kept as `--head pair`, scores an action from what is known about its item
and about its target cell (category, place, entity standing there). What is learned about a
cell then serves every item. One run each, with one known build on both sides, against an
enemy that only ends its turns, 60 updates: 100% of wins in 5.8 turns for pairs, 87% in
23.7 turns for independent outputs. In its present form it can hardly choose among several
targets from a combination of circumstances, which placement items (inversion) need. A
small network looking at item, target and situation together would allow it.

## Both sides of the map must be seen in training

Date: 2026-11-24. Status: accepted.

A network trained only as team 0 won 256 fights out of 256 as team 0, and drew 97% as team
1. The likely reason, not tested on its own, is that the same situation facing the other
way looks different. Self-play shows both sides; opponents that do not learn alternate.
Mirroring the observation is an idea for later.
