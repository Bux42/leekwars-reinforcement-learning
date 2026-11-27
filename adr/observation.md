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
masked, except summons and teleports. 518,000 masked casts were executed on cloned fights,
none had an effect. 84% of legal casts are masked.

Some casts matter to their caster whatever they reach: a `sword`'s shield, `enhanced_lightninger`'s
heal, critical hits of a caster they would heal (`unstable_destroyer`, with life missing, not
unhealable and some agility). Cast on nobody, they all do the same thing, wherever: only one
of them is kept, for each item (2026-11-26). Before that, a hurt leek with
`unstable_destroyer` had 347 legal actions instead of about 65, 29% of its casts reached
nobody, and it scored 41% against random builds instead of 51%. With the same network and
fights, it scores 50% after the change. 1 million masked casts were checked: each did
exactly what the kept one of its item does.

What `j_laser` and `quantum_rifle` steal for their caster depends on who they reach: they
are masked too when they reach nobody.

Masked too when it reaches someone, but nobody it can change:

| Effect | Nothing to change when |
|---|---|
| Instant heal | no life is missing, or the target is unhealable. Heals over time are kept |
| Loss of maximum life: `alteration`, `desintegration` | the target is at full life: at most the life missing is taken |
| `superinfection` | it carries no poison |
| `antidote` | it carries no poison, and lacks no life |
| `liberation` | it carries no effect, or only irreducible ones |
| `covid` | it already carries it |

Found by executing the kept casts on clones, 8 times each with other random numbers: 2.4% never
changed anything, before the last five were masked. What remains is damage that shields absorb,
and values that round down to 0 (`rage`, `soporific`...). Both depend on characteristics and
rolls, the agent can learn them.

## Harmful casts are masked

Date: 2026-11-24. Status: accepted, after a review item by item. `--trim useless` goes
without.

Casts that can only harm their own team, or only help the enemy, are masked. One that has
anything else to offer stays, like a `tranquilizer` that reaches an ally and an enemy.

| Masked | Items |
|---|---|
| It only reaches its team, and only takes TP, MP or characteristics from it, or makes it unhealable | `tranquilizer`, `soporific`, `slow_down`, `ball_and_chain`, `fracture`, `crushing`, `brainwashing`, `hemorrhage`, `plutonium_bazooka` |
| It only reaches enemies (for an instant heal, those that lack life), and only gives them life, maximum life, TP or characteristics, or removes their shackles | `bandage`, `cure`, `drip`, `therapy`, `regeneration`, `explorer_rifle`, `armoring`, `elevation`, `mutation`, `transmutation`, `knowledge`, `prism`, `wizardry`, `manumission` |
| It only harms its caster, a leek | `shock`, `pebble`, `ice`, `spark`, `flash`, `devil_strike`, `plasma`, `toxin`, `plague`, `gazor`, `desintegration` on itself; `punishment`, `dark_katana` on an empty cell |

Such a cast can still be a strategy. It is kept in these cases:

| Kept | Why |
|---|---|
| The caster has `unstable_destroyer`, lacks life, is not unhealable and has agility | Its critical hits heal it, whatever it casts. Of the casts that reach nobody, one per item is kept |
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
The world has since been turned to face the enemy, which probably makes this pointless: it
has not been checked.

## The world is turned to face the enemy

Date: 2026-11-25. Status: accepted. `--no-face-enemy` goes without.

Each output stands for a move or a cast at a relative cell. "Walk towards the enemy" was
therefore a different set of outputs for each direction, learned separately: agents stopped
about 20 cells away from the enemy, as untrained ones do, and half of the fights were draws.

The world is now turned, by one of the 8 symmetries of the grid, so that the enemy leek is
always in the same eighth of the directions. No reward and no rule changes, and no action is
lost: the same game is seen turned (checked decision by decision, on the same fights). Only
the points of the compass disappear, which mean nothing in the rules.

1,000 updates each, same settings but for this and, for two of them, the next section, then
1,000 to 2,000 fights: wins against an enemy that only ends its turns, 28% without, 75 to 76%
with, for three seeds. Against the network trained without, 56 to 59% wins and 8 to 9%
losses. Fights became shorter by themselves: 20% of draws instead of 49%, 340 decisions per
fight instead of 880.

What it assumes: nothing in the engine prefers a direction. Details might, such as the path
chosen among equally short ones.

## No move right after a move

Date: 2026-11-25. Status: accepted. `--repeat-moves` goes without.

A fifth of all decisions were moves of an entity that had just moved. A single move always
reaches the same cell for as few MP: checked on 6.2 million pairs of moves, 63% of which
wasted MP. So an entity that has just moved must cast before moving again.

A leek then takes 3.8 decisions per turn instead of 4.5. But agents summon and cast more: a
fight still takes 300 to 350 decisions, and they are neither better nor worse (41% of wins
against 40 to 43% of losses, against a network trained without this rule).

It can move again after a move whose path could have entered the zone of a plant that it has
not awakened yet: the path then matters. Also after a move made with more than 20 MP, the
longest move.

## Cells from which the enemy leek can be seen

Date: 2026-11-26. Status: accepted, the default since 2026-11-27. `--no-sight` goes without.

The network gets the cells it could move to this turn, its own included, from which the
enemy leek would be in its line of sight. It follows from the map: given, not guessed, so
that learning where to go does not wait for learning line of sight. Range and the rules of
first-in-line items are left out. About 4% slower training.

With the same settings otherwise, 10,000 updates each, 1,500 to 2,000 fights per pair:
12 to 17 Elo more than without it from update 2,000 on, with the small network (256 x 2)
and with a 4 times wider one (1024 x 2) alike. The wider network itself gave at most 4 to 8
Elo, 30% slower: capacity was not what held the agent back.

## Actions see their target

Date: 2026-11-27. Status: being confirmed on a full run. `--head target`.

With one output per action, each of the 10,566 actions learns on its own what it targets.
The target head scores each legal action with a small network (128 wide) that sees the
situation, the tool (end of turn, move, weapon or chip) and facts about the target cell:
its category, place, the entity on it, and whether the enemy leek can be seen from it. A
bias per action keeps what is particular to each. It is told nothing about what is good.

2,500 updates, otherwise as with one output per action and sight: update 1,000 is as strong
as update 5,000 of one output per action, update 2,500 beats update 10,000 (48.7% of wins
against 44.9%) and the 4 times wider network (48.3% against 45.7%). It plays 6% of draws
instead of 9 to 10%. 35% slower, so about 3 times more learned per hour.
