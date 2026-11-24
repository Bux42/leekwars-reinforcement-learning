# Rewards (1v1, first version)

Accepted for the first version (2026-11-24), coded in
[reward.h](../include/leekrl/reward.h). All numbers are settings, and each part is kept
separate so its influence can be seen.

## What the engine reports

Measured on 1,000 random fights between random builds, through the generator's
statistics callbacks. They work: for the leeks, damage minus heals matches the life
really lost, to the point (1,429,622).

| Damage type | Removes | Reported when | Share of life lost by leeks |
|---|---|---|---|
| Direct | Life | Cast | 58% |
| Poison | Life | Start of the target's turns | 32% |
| Life (a share of the caster's current life) | Life | Cast | 9% |
| Return | Life | Cast, on the attacker | 1% |
| Aftereffect | Life | Cast, then start of the target's turns | Only on summons here |
| Nova | Total life | Cast | - |
| Erosion | Total life | With any other damage | - |

- **Nova and erosion share one type.** Only the effect tells them apart. Erosion is about
  four times larger in total.
- **Healing undoes a lot:** leeks healed the equivalent of 55% of the damage they took.
- **43% of all damage was self-inflicted** (leeks and summons), since random play does not
  aim. A new agent will do the same.
- Summons dying with their summoner are reported without a killer.

## Two ways to reward damage

**A. Count events**, as in the first spec: a reward for each point of damage dealt, a
penalty for each point received, fixed rewards for kills.

**B. Compare situations** (chosen): give each team a score from the current life of
its entities. The reward is how much the score difference moved since the team's
previous decision.

| | A. Events | B. Situations |
|---|---|---|
| Damage, whatever its type | One rule per type | Automatic |
| Healing, vitality, shields | Extra rules, or ignored | Automatic, shields through damage avoided |
| Damage then heal, repeated | Can earn reward for nothing | Sums to zero |
| Summoning | Penalized: own summon's death costs, summoning gives nothing | Neutral: what it gives is lost when it dies |
| Poison | When it ticks, or when cast (to choose) | When it ticks |
| Engine callbacks needed | Yes | No |

With B, an agent cannot collect reward by going around in circles, which is the usual
trap of reward shaping.

## Proposed rewards

### Score of a team

```
leek:         0.30 x life / L  +  0.10 x total life / L
each summon:  0.01  +  0.01 x life / total life
```

`L` is the leek's total life at the start of the fight, so a leek with 500 life and one
with 5,000 weigh the same. Dead entities count 0. Total life is included so that nova
and erosion count, for less than life itself.

### Reward at each decision

```
D = score of my team - score of the enemy team
reward = D now - D at my team's previous decision
```

It covers everything that happened in between: my action, the enemy's whole turn, poison,
plants. Over a whole fight, the two sides' totals are opposite.

These rewards add up to the final `D` minus the initial one (0), so about -0.56 to +0.56
whatever happened in the middle: 0.40 for a leek at full life and 0.16 for 8 summons.
Vitality effects, which raise total life, can exceed it: the largest difference at the
end of 1,000 random fights was 0.89.

### End of the fight

| Outcome | Reward |
|---|---|
| Win | +1 |
| Loss | -1 |
| Draw (64 turns) | -0.2 for both |
| Win: shorter fight | + 0.2 x (1 - turn / 64) |
| Win: fewer actions | + 0.1 x 100 / (100 + actions) |

Draws are slightly negative so that two cautious agents do not settle for them.

**Actions** are all movements and casts of the winner's team during the fight, summons
included, ends of turn excluded. Between one cast of 8 TP for 100 damage and two casts of
4 TP for 50 each, the single cast is favored. The bonus is a tie-breaker: 50 actions
instead of 100 give 0.017 more. It also counts against playing summons or several cheap
chips, so it should stay small. If it has no visible effect, the next thing to try is a
tiny cost per action, which the agent would feel at once.

### Example

Two leeks, 2,000 life each. I deal 500 damage with 50 erosion, the enemy then heals 200.

```
enemy score: 0.30 x 1500/2000 + 0.10 x 1950/2000 = 0.3225   (was 0.40)
after heal:  0.30 x 1700/2000 + 0.10 x 1950/2000 = 0.3525
reward before the enemy plays:   +0.0775
reward at my next decision:      -0.0300
```

## How the first spec maps to this

| First spec | Here |
|---|---|
| Big reward for winning, big loss for losing | +1 / -1 |
| Shorter fights | Duration bonus |
| Fewer total actions | Actions bonus |
| Killing an entity, smaller for summons | Win in 1v1; a summon is worth 0.01 to 0.02 |
| Ally killed | Same, negative |
| Damage dealt and received, smaller for summons | Score changes |

In team fights, killing a leek would need its own fixed part in the score.

## Notes for training

- **Discount:** random fights take about 650 decisions, both teams together. A discount
  of 0.999 keeps the end of the fight visible. 0.99 would make it vanish.
- **Order of magnitude:** winning must stay above anything else. The score alone usually
  cannot make up for the gap between winning and losing (2).
- **Ablations to run:** no damage reward at all (only the end of the fight), and weights
  halved or doubled.
- **Poison is rewarded when it ticks**, not when cast. If the agent fails to learn poison
  builds, crediting the expected damage when cast is the next thing to try.

## Checks

In [test_reward.c](../tests/test_reward.c):

1. The two sides' totals over a fight are opposite, end rewards aside.
2. Rewards of a whole fight add up to final minus initial difference, plus the end.
3. Damage then an equal heal gives zero.
4. Each damage type of the table moves the score as expected.
5. Summoning then losing the summon gives zero.
6. End rewards: win, loss, draw, duration, actions.
