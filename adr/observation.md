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

### Alternatives

- Sum of amount x remaining duration as a third float (total future value).
- Amounts per remaining duration: 1 turn, 2 turns, 3 turns, 4 and more.
- A fixed number of slots per entity, one per effect. Complete until it
  overflows, but larger, and the meaning of a slot changes.

### Signs it should be revisited

- Wasted healing or antidotes against poison, or deaths to foreseeable poison.
- Attacks made just before a shield expires instead of just after.
- Poor timing of buffs, such as renewing a buff that is mostly still active.
