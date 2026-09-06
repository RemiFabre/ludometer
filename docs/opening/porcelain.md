# Opening atlas: porcelain at 4,096 sims

*Generated from `data/cloud/opening/porcelain_r01_s4096.npz`: 86,613 positions of the expert games in rounds 1-2 (both seats), 45,876 of them the expert's (the dataset's target seat). Search: no root noise, 4,096 simulations per position, c_puct 1.4, 4 determinizations per refill edge. Labelled in 141 min on mps.*

Columns: **top-1** = the move the net would play (decisive pick) equals the expert's; **top-1 (visits)** = most-visited move equals the expert's; **top-3** = the expert's move is among the three most visited; **raw prior top-1** = the policy head alone, no search; **expert visit share** = the fraction of the search that went into the expert's move; **loss** = Q(net's move) - Q(expert's move) in the mover's frame, win-probability units on [-1, 1] (0.10 ≈ 5 percentage points of win chance); **pts** = the same gap read from the margin head, in final-score points; **unvisited** = the search never tried the expert's move (the loss is then unknown and excluded).

## 1. Agreement by move

**Round 1** (the target seat, i.e. the expert; the last row is the other seat for contrast)

| slice | n | top-1 | top-1 (visits) | top-3 | raw prior top-1 | expert visit share | mean loss | median loss | p90 loss | loss>0.05 | loss>0.10 | loss>0.20 | mean pts | unvisited |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round 1, move 1 | 4,030 | 29.3% | 28.8% | 55.6% | 23.3% | 26.9% | 0.086 | 0.030 | 0.267 | 44.2% | 32.3% | 16.5% | +0.91 | 14.4% |
| round 1, move 2 | 4,030 | 36.2% | 38.7% | 65.2% | 33.2% | 34.6% | 0.072 | 0.015 | 0.232 | 37.0% | 26.2% | 13.0% | +0.85 | 7.1% |
| round 1, move 3 | 4,030 | 38.6% | 42.2% | 71.5% | 37.9% | 38.0% | 0.068 | 0.012 | 0.228 | 35.6% | 23.9% | 12.2% | +0.85 | 4.0% |
| round 1, move 4 | 4,030 | 49.2% | 50.6% | 79.2% | 44.2% | 47.5% | 0.063 | 0.000 | 0.212 | 33.2% | 22.6% | 10.9% | +0.84 | 2.4% |
| round 1, move 5 | 3,974 | 66.8% | 68.0% | 91.8% | 64.4% | 65.8% | 0.039 | 0.000 | 0.143 | 20.8% | 13.7% | 6.8% | +0.58 | 0.5% |
| round 1, move 6 | 2,706 | 79.9% | 81.4% | 97.9% | 80.6% | 79.7% | 0.019 | 0.000 | 0.056 | 10.7% | 6.5% | 2.5% | +0.30 | 2.7% |
| round 1, move 7 | 629 | 87.6% | 89.3% | 99.0% | 89.3% | 87.3% | 0.014 | 0.000 | 0.020 | 7.4% | 4.6% | 1.8% | +0.23 | 10.0% |
| round 1, move 8 | 49 | 89.8% | 89.8% | 100.0% | 91.8% | 87.0% | 0.024 | 0.000 | 0.129 | 12.5% | 12.5% | 2.5% | +0.40 | 18.4% |
| round 1, moves 1-3 | 12,090 | 34.7% | 36.6% | 64.1% | 31.4% | 33.2% | 0.075 | 0.018 | 0.242 | 38.8% | 27.3% | 13.8% | +0.87 | 8.5% |
| round 1, all | 23,479 | 49.4% | 51.0% | 76.3% | 46.6% | 48.1% | 0.058 | 0.000 | 0.206 | 30.4% | 21.0% | 10.4% | +0.73 | 5.5% |
| round 1, opponent seat | 20,729 | 31.2% | 32.5% | 57.4% | 31.6% | 30.8% | 0.121 | 0.053 | 0.351 | 50.7% | 40.1% | 24.0% | +1.68 | 9.1% |

**Round 1, the disagreements re-searched from the child positions** (a full 4,096-sim search after the expert's move and after the net's; round-ending moves excluded)

| slice | disagreements searched | root-Q loss (same rows) | child-search loss, mean | median | child loss > 0.10 | expert's child better (loss < -0.02) |
|---|---|---|---|---|---|---|
| round 1, move 1 | 2,673 | 0.130 | 0.154 | 0.113 | 53.1% | 12.8% |
| round 1, move 2 | 2,423 | 0.115 | 0.104 | 0.077 | 43.7% | 19.9% |
| round 1, move 3 | 2,325 | 0.114 | 0.097 | 0.074 | 42.5% | 20.1% |
| round 1, move 4 | 1,931 | 0.127 | 0.102 | 0.070 | 41.7% | 19.6% |
| round 1, move 5 | 1,223 | 0.118 | 0.078 | 0.050 | 35.5% | 22.6% |
| round 1, move 6 | 464 | 0.087 | 0.044 | 0.024 | 25.6% | 25.9% |
| round 1, move 7 | 50 | 0.095 | 0.044 | -0.003 | 36.0% | 34.0% |
| round 1, move 8 | 3 | 0.218 | 0.042 | -0.052 | 33.3% | 66.7% |
| round 1, moves 1-3 | 7,421 | 0.120 | 0.120 | 0.089 | 46.7% | 17.4% |
| round 1, all | 11,092 | 0.120 | 0.109 | 0.077 | 43.7% | 18.8% |

**Round 2** (the target seat, i.e. the expert; the last row is the other seat for contrast)

| slice | n | top-1 | top-1 (visits) | top-3 | raw prior top-1 | expert visit share | mean loss | median loss | p90 loss | loss>0.05 | loss>0.10 | loss>0.20 | mean pts | unvisited |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round 2, move 1 | 4,030 | 37.5% | 43.6% | 68.5% | 41.9% | 37.4% | 0.047 | 0.007 | 0.162 | 25.5% | 16.3% | 6.8% | +0.70 | 5.5% |
| round 2, move 2 | 4,030 | 34.1% | 38.9% | 66.1% | 36.2% | 32.6% | 0.059 | 0.012 | 0.190 | 31.5% | 20.3% | 9.4% | +0.84 | 5.0% |
| round 2, move 3 | 4,030 | 42.7% | 44.6% | 74.5% | 38.7% | 39.9% | 0.066 | 0.006 | 0.213 | 32.8% | 22.2% | 10.7% | +0.98 | 3.0% |
| round 2, move 4 | 4,030 | 56.5% | 57.7% | 86.5% | 46.9% | 53.6% | 0.055 | 0.000 | 0.193 | 26.4% | 18.2% | 9.6% | +0.88 | 1.2% |
| round 2, move 5 | 3,829 | 73.0% | 74.2% | 95.6% | 67.4% | 70.6% | 0.031 | 0.000 | 0.104 | 14.8% | 10.3% | 5.4% | +0.52 | 2.5% |
| round 2, move 6 | 2,047 | 81.3% | 82.5% | 98.2% | 79.6% | 79.8% | 0.021 | 0.000 | 0.059 | 10.7% | 7.3% | 3.1% | +0.36 | 13.6% |
| round 2, move 7 | 375 | 82.7% | 83.2% | 99.2% | 85.6% | 82.0% | 0.021 | 0.000 | 0.054 | 10.9% | 6.2% | 2.9% | +0.34 | 26.9% |
| round 2, moves 1-3 | 12,090 | 38.1% | 42.4% | 69.7% | 39.0% | 36.6% | 0.057 | 0.008 | 0.186 | 30.0% | 19.6% | 9.0% | +0.84 | 4.5% |
| round 2, all | 22,397 | 52.1% | 55.0% | 80.3% | 49.8% | 50.2% | 0.049 | 0.000 | 0.169 | 24.8% | 16.5% | 7.9% | +0.74 | 4.8% |
| round 2, opponent seat | 20,008 | 37.9% | 41.4% | 70.3% | 44.1% | 37.6% | 0.075 | 0.019 | 0.232 | 36.8% | 24.8% | 12.4% | +1.34 | 5.2% |

**Round 1 by the expert's Elo** (the source ladder's raw scale; the pipeline's floor is 1950)

| slice | n | top-1 | top-1 (visits) | top-3 | raw prior top-1 | expert visit share | mean loss | median loss | p90 loss | loss>0.05 | loss>0.10 | loss>0.20 | mean pts | unvisited |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| expert Elo < 2200 | 2,067 | 46.3% | 48.3% | 72.9% | 45.4% | 45.6% | 0.072 | 0.000 | 0.238 | 35.9% | 25.9% | 12.8% | +0.88 | 5.0% |
| 2200-2350 | 1,066 | 54.2% | 55.8% | 80.1% | 50.8% | 51.9% | 0.051 | 0.000 | 0.185 | 25.9% | 18.1% | 8.6% | +0.69 | 4.5% |
| ≥ 2350 | 20,346 | 49.4% | 51.0% | 76.4% | 46.5% | 48.1% | 0.057 | 0.000 | 0.203 | 30.1% | 20.6% | 10.2% | +0.72 | 5.6% |

## 2. How much the net thinks the expert lost

**Round 1, expert's decisions**

| quantile | value loss | points |
|---|---|---|
| p50 | 0.000 | +0.00 |
| p75 | 0.077 | +1.10 |
| p90 | 0.206 | +2.53 |
| p95 | 0.290 | +3.46 |
| p99 | 0.454 | +5.54 |
| mean | 0.058 | +0.73 |

Disagreements: 48.4% of decisions. Among them the mean value loss is 0.120 (+1.51 points); 20.0% are within 0.02 of the net's own pick (a coin flip to the net), 43.3% are what the net calls a clear mistake (> 0.10), 21.4% a blunder (> 0.20).

**Rounds 1-2, expert's decisions**

| quantile | value loss | points |
|---|---|---|
| p50 | 0.000 | +0.00 |
| p75 | 0.062 | +1.03 |
| p90 | 0.187 | +2.54 |
| p95 | 0.280 | +3.54 |
| p99 | 0.464 | +5.77 |
| mean | 0.054 | +0.74 |

Disagreements: 47.9% of decisions. Among them the mean value loss is 0.112 (+1.54 points); 24.1% are within 0.02 of the net's own pick (a coin flip to the net), 39.2% are what the net calls a clear mistake (> 0.10), 19.1% a blunder (> 0.20).

## 3. What the disagreements look like (round 1)

Disagreements with value loss > 0.05 (target seat): 6,743. Coarse patterns (a disagreement can count in two rows):

| pattern | count | share |
|---|---|---|
| different colour | 4892 | 72.5% |
| center vs factory: expert center | 1391 | 20.6% |
| same colour, other row (expert lower row) | 720 | 10.7% |
| center vs factory: net center | 717 | 10.6% |
| same colour, other row (expert higher row) | 452 | 6.7% |
| same colour and row, other source | 379 | 5.6% |
| expert builds, net floors | 156 | 2.3% |
| expert floors, net builds | 144 | 2.1% |

Finest patterns, top 12:

| pattern | count |
|---|---|
| different colour; same row | 470 |
| different colour; different factory; row 3 vs row 4 (expert higher) | 187 |
| different colour; different factory; same row | 184 |
| same colour; different factory; same row | 163 |
| same colour; row 5 vs row 4 (expert lower) | 142 |
| different colour; different factory; row 3 vs row 5 (expert higher) | 132 |
| different colour; different factory; row 5 vs row 4 (expert lower) | 130 |
| different colour; expert center, net factory; same row | 129 |
| same colour; expert center, net factory; same row | 126 |
| same colour; row 5 vs row 2 (expert lower) | 118 |
| same colour; row 4 vs row 3 (expert lower) | 116 |
| different colour; expert center, net factory; row 1 vs row 4 (expert higher) | 107 |

## 4. The 20 largest round-1 disagreements

Boards are printed before the move, from the engine's text renderer: each player's five pattern lines on the left (`.` empty slot), the wall on the right (`.` empty), colours B=blue Y=yellow R=red K=black T=teal, `#` on the floor = the first-player marker; `*` marks the player to move.

### 1. Round 1, the expert's move 4 (game 1010, position 6; expert Elo 2486)

- **Expert took** 2 red from the center to the 2-tile row, completing it. Net's Q for it -0.222, 0.2% of visits, raw prior 8.3%.
- **Net wanted** 2 yellow from factory 1 to the 3-tile row (already 1/3), completing it. Q +0.524, 99.0% of visits, raw prior 58.9%.
- Searched from the children: after the expert's move -0.530, after the net's +0.566, **child-search loss 1.096**.
- Root-edge value loss **0.746** (+8.9 points by the margin head); root value +0.516. The expert won the game 26-12.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: -
  1: YYRT
  2: -
  3: -
  4: -
  center: BRRKTT
  bag: B17 Y13 R16 K17 T17   lid: B0 Y0 R0 K0 T0
*P0  score 0
        K | .....
       .. | .....
      ..Y | .....
     ..BB | .....
    ..... | .....
    floor: K (-1)
 P1  score 0
        R | .....
       YY | .....
      ... | .....
     ..YY | .....
    ..... | .....
    floor: # (-1)
```

### 2. Round 1, the expert's move 6 (game 1340, position 10; expert Elo 2460)

- **Expert took** 3 red from the center to the 3-tile row, completing it. Net's Q for it -0.291, 0.1% of visits, raw prior 5.2%.
- **Net wanted** 2 teal from the center to the 3-tile row. Q +0.491, 97.1% of visits, raw prior 31.0%.
- Searched from the children: after the expert's move -0.473, after the net's +0.521, **child-search loss 0.994**.
- Root-edge value loss **0.782** (+8.8 points by the margin head); root value +0.476. The expert won the game 62-38.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: BRRRTT
  bag: B16 Y16 R15 K19 T14   lid: B0 Y0 R0 K0 T0
*P0  score 0
        Y | .....
       RR | .....
      ... | .....
     TTTT | .....
    ..... | .....
    floor: # (-1)
 P1  score 0
        Y | .....
       YY | .....
      BBB | .....
     .... | .....
    ....K | .....
    floor: - (0)
```

### 3. Round 1, the expert's move 4 (game 616, position 6; expert Elo 2158)

- **Expert took** 2 yellow from the center to the 4-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.1%.
- **Net wanted** 1 black from factory 2 to the 5-tile row. Q +0.409, 55.8% of visits, raw prior 20.0%.
- Searched from the children: after the expert's move -0.569, after the net's +0.398, **child-search loss 0.967**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.379. The expert lost the game 49-8.

```
round 0  to move: P1  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: BYKT
  3: -
  4: -
  center: BYYR
  bag: B13 Y17 R15 K18 T17   lid: B0 Y0 R0 K0 T0
 P0  score 0
        . | .....
       RR | .....
      ... | .....
     ..BB | .....
    ...TT | .....
    floor: - (0)
*P1  score 0
        K | .....
       RR | .....
      BBB | .....
     .... | .....
    ..... | .....
    floor: # (-1)
```

### 4. Round 1, the expert's move 5 (game 3479, position 8; expert Elo 2460)

- **Expert took** 1 red from the center to the 4-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.4%.
- **Net wanted** 2 black from factory 1 to the 4-tile row. Q +0.480, 95.1% of visits, raw prior 34.4%.
- Searched from the children: after the expert's move -0.435, after the net's +0.521, **child-search loss 0.957**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.467. The expert won the game 74-23.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: -
  1: BRKK
  2: -
  3: -
  4: -
  center: RK
  bag: B14 Y15 R16 K16 T19   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       BB | .....
      BBB | .....
     .... | .....
    .YYYY | .....
    floor: - (0)
 P1  score 0
        K | .....
       RR | .....
      ... | .....
     .... | .....
    ....T | .....
    floor: Y# (-2)
```

### 5. Round 1, the expert's move 4 (game 2319, position 7; expert Elo 2460)

- **Expert took** 2 blue from the center to the 4-tile row. Net's Q for it -0.118, 0.0% of visits, raw prior 1.0%.
- **Net wanted** 1 yellow from the center to the 2-tile row (already 1/2), completing it. Q +0.477, 96.0% of visits, raw prior 44.1%.
- Searched from the children: after the expert's move -0.421, after the net's +0.488, **child-search loss 0.909**.
- Root-edge value loss **0.596** (+9.1 points by the margin head); root value +0.466. The expert won the game 85-39.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: -
  3: YRTT
  4: -
  center: BBYT
  bag: B16 Y14 R13 K20 T17   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .Y | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        R | .....
       RR | .....
      RRR | .....
     .... | .....
    ...BB | .....
    floor: # (-1)
```

### 6. Round 1, the expert's move 4 (game 2518, position 6; expert Elo 2460)

- **Expert took** 1 red from factory 1 to the 4-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.1%.
- **Net wanted** 1 red from factory 1 to the 1-tile row, completing it. Q +0.127, 89.7% of visits, raw prior 16.0%.
- Searched from the children: after the expert's move -0.709, after the net's +0.184, **child-search loss 0.893**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.108. The expert won the game 74-40.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: -
  1: BYRK
  2: -
  3: -
  4: -
  center: BYKKT
  bag: B16 Y11 R17 K17 T19   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       BB | .....
      ... | .....
     .... | .....
    YYYYY | .....
    floor: - (0)
 P1  score 0
        . | .....
       RR | .....
      .YY | .....
     .... | .....
    ..... | .....
    floor: # (-1)
```

### 7. Round 1, the expert's move 4 (game 859, position 7; expert Elo 2486)

- **Expert took** 2 teal from the center to the 4-tile row (already 2/4), completing it. Net's Q for it -0.123, 0.5% of visits, raw prior 11.7%.
- **Net wanted** 4 black from the center to the 5-tile row. Q +0.360, 99.1% of visits, raw prior 76.2%.
- Searched from the children: after the expert's move -0.493, after the net's +0.395, **child-search loss 0.888**.
- Root-edge value loss **0.484** (+5.0 points by the margin head); root value +0.356. The expert lost the game 40-50.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: YKKKKTT
  bag: B18 Y15 R18 K15 T14   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       TT | .....
      .BB | .....
     ..TT | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        K | .....
       RR | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: Y# (-2)
```

### 8. Round 1, the expert's move 1 (game 1922, position 0; expert Elo 2019)

- **Expert took** 2 yellow from factory 2 to the 3-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.0%.
- **Net wanted** 2 teal from factory 4 to the 4-tile row. Q +0.353, 48.1% of visits, raw prior 2.3%.
- Searched from the children: after the expert's move -0.497, after the net's +0.380, **child-search loss 0.878**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.297. The expert lost the game 57-55.

```
round 0  to move: P1  first player: P1  scores: 0-0
Factories:
  0: YYKT
  1: YRKK
  2: BYYT
  3: BBYR
  4: BBTT
  center: - [1st]
  bag: B15 Y14 R18 K17 T16   lid: B0 Y0 R0 K0 T0
 P0  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
*P1  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 9. Round 1, the expert's move 4 (game 23, position 6; expert Elo 2486)

- **Expert took** 1 teal from factory 2 to the 5-tile row (already 3/5). Net's Q for it -0.387, 0.3% of visits, raw prior 9.4%.
- **Net wanted** 1 teal from factory 2 to the 1-tile row, completing it. Q +0.252, 78.3% of visits, raw prior 44.0%.
- Searched from the children: after the expert's move -0.452, after the net's +0.414, **child-search loss 0.866**.
- Root-edge value loss **0.639** (+6.2 points by the margin head); root value +0.242. The expert won the game 47-35.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: -
  1: -
  2: YYYT
  3: -
  4: -
  center: -
  bag: B16 Y14 R18 K16 T16   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       BB | .....
      YYY | .....
     .... | .....
    ..TTT | .....
    floor: - (0)
 P1  score 0
        . | .....
       RR | .....
      ... | .....
     ..BB | .....
    .KKKK | .....
    floor: # (-1)
```

### 10. Round 1, the expert's move 3 (game 118, position 5; expert Elo 2486)

- **Expert took** 3 teal from the center to the 5-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.2%.
- **Net wanted** 3 teal from the center to the 3-tile row, completing it. Q +0.529, 96.8% of visits, raw prior 55.6%.
- Searched from the children: after the expert's move -0.353, after the net's +0.512, **child-search loss 0.865**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.520. The expert won the game 36-10.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: -
  1: YYRK
  2: -
  3: -
  4: -
  center: BYYKKTTT
  bag: B17 Y16 R16 K14 T17   lid: B0 Y0 R0 K0 T0
*P0  score 0
        R | .....
       RR | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: # (-1)
 P1  score 0
        . | .....
       BB | .....
      KKK | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 11. Round 1, the expert's move 4 (game 2976, position 7; expert Elo 2460)

- **Expert took** 3 blue from the center to the 4-tile row. Net's Q for it -0.158, 0.4% of visits, raw prior 12.9%.
- **Net wanted** 2 teal from the center to the 4-tile row. Q +0.438, 97.9% of visits, raw prior 66.7%.
- Searched from the children: after the expert's move -0.436, after the net's +0.428, **child-search loss 0.865**.
- Root-edge value loss **0.596** (+6.2 points by the margin head); root value +0.431. The expert won the game 53-40.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: BBBYRTT
  bag: B17 Y13 R16 K17 T17   lid: B0 Y0 R0 K0 T0
*P0  score 0
        K | .....
       YY | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: Y (-1)
 P1  score 0
        T | .....
       KK | .....
      RRR | .....
     .... | .....
    ..... | .....
    floor: # (-1)
```

### 12. Round 1, the expert's move 1 (game 1140, position 1; expert Elo 2460)

- **Expert took** 1 red from the center to the 4-tile row (takes the first-player marker). Net's Q for it +nan, 0.0% of visits, raw prior 0.2%.
- **Net wanted** 1 red from the center to the 5-tile row (takes the first-player marker). Q +0.251, 8.0% of visits, raw prior 2.0%.
- Searched from the children: after the expert's move -0.692, after the net's +0.166, **child-search loss 0.858**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.237. The expert lost the game 41-48.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: BRKT
  1: YRKT
  2: YRRT
  3: BBRT
  4: -
  center: RK [1st]
  bag: B17 Y16 R14 K17 T16   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      .YY | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 13. Round 1, the expert's move 3 (game 1135, position 5; expert Elo 2460)

- **Expert took** 2 yellow from factory 0 to the 2-tile row, completing it. Net's Q for it -0.279, 0.1% of visits, raw prior 2.5%.
- **Net wanted** 2 black from the center to the 5-tile row. Q +0.176, 98.7% of visits, raw prior 73.5%.
- Searched from the children: after the expert's move -0.564, after the net's +0.284, **child-search loss 0.848**.
- Root-edge value loss **0.455** (+5.0 points by the margin head); root value +0.170. The expert won the game 61-58.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: BYYK
  1: -
  2: -
  3: -
  4: -
  center: BBYYRRKK
  bag: B15 Y16 R17 K17 T15   lid: B0 Y0 R0 K0 T0
*P0  score 0
        R | .....
       .. | .....
      ... | .....
     ..TT | .....
    ..... | .....
    floor: # (-1)
 P1  score 0
        . | .....
       BB | .....
      TTT | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 14. Round 1, the expert's move 2 (game 3566, position 2; expert Elo 2460)

- **Expert took** 3 red from factory 1 to the 4-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.0%.
- **Net wanted** 3 red from factory 1 to the 5-tile row. Q +0.341, 13.7% of visits, raw prior 14.8%.
- Searched from the children: after the expert's move -0.485, after the net's +0.353, **child-search loss 0.839**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.345. The expert won the game 62-39.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: RKTT
  1: RRRT
  2: YYYT
  3: -
  4: -
  center: B [1st]
  bag: B19 Y13 R16 K16 T16   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      KKK | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      ... | .....
     YYYY | .....
    ..... | .....
    floor: - (0)
```

### 15. Round 1, the expert's move 1 (game 418, position 0; expert Elo 2486)

- **Expert took** 2 blue from factory 1 to the 4-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.0%.
- **Net wanted** 2 black from factory 0 to the 5-tile row. Q +0.308, 93.8% of visits, raw prior 70.8%.
- Searched from the children: after the expert's move -0.522, after the net's +0.310, **child-search loss 0.833**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.303. The expert won the game 62-46.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: YYKK
  1: BBRT
  2: YYYR
  3: YRKT
  4: YRRT
  center: - [1st]
  bag: B18 Y13 R15 K17 T17   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 16. Round 1, the expert's move 1 (game 1010, position 0; expert Elo 2486)

- **Expert took** 2 blue from factory 2 to the 4-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.0%.
- **Net wanted** 2 black from factory 4 to the 5-tile row. Q +0.316, 92.7% of visits, raw prior 82.9%.
- Searched from the children: after the expert's move -0.508, after the net's +0.317, **child-search loss 0.825**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.313. The expert won the game 26-12.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: BYKT
  1: YYRT
  2: BBYR
  3: YYRT
  4: YRKK
  center: - [1st]
  bag: B17 Y13 R16 K17 T17   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 17. Round 1, the expert's move 3 (game 3410, position 5; expert Elo 2460)

- **Expert took** 3 yellow from the center to the 4-tile row. Net's Q for it -0.720, 0.0% of visits, raw prior 1.1%.
- **Net wanted** 1 red from factory 4 to the 1-tile row, completing it. Q +0.004, 97.8% of visits, raw prior 52.9%.
- Searched from the children: after the expert's move -0.691, after the net's +0.123, **child-search loss 0.814**.
- Root-edge value loss **0.724** (+10.2 points by the margin head); root value -0.006. The expert won the game 63-46.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: BYRK
  center: BYYYK
  bag: B18 Y13 R17 K14 T18   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       RR | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       TT | .....
      ... | .....
     KKKK | .....
    ..... | .....
    floor: # (-1)
```

### 18. Round 1, the expert's move 1 (game 2896, position 0; expert Elo 2460)

- **Expert took** 2 blue from factory 0 to the 4-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.0%.
- **Net wanted** 2 teal from factory 2 to the 4-tile row. Q +0.307, 98.1% of visits, raw prior 84.4%.
- Searched from the children: after the expert's move -0.521, after the net's +0.290, **child-search loss 0.810**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.304. The expert won the game 65-30.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: BBRT
  1: BBRR
  2: YKTT
  3: BYRT
  4: BRRK
  center: - [1st]
  bag: B14 Y18 R14 K18 T16   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 19. Round 1, the expert's move 3 (game 2110, position 5; expert Elo 2460)

- **Expert took** 2 blue from the center to the 4-tile row. Net's Q for it -0.381, 0.1% of visits, raw prior 3.2%.
- **Net wanted** 1 yellow from the center to the 3-tile row (already 2/3), completing it. Q +0.321, 98.7% of visits, raw prior 60.8%.
- Searched from the children: after the expert's move -0.465, after the net's +0.343, **child-search loss 0.808**.
- Root-edge value loss **0.702** (+7.0 points by the margin head); root value +0.312. The expert won the game 55-45.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: BYRK
  1: -
  2: -
  3: -
  4: -
  center: BBYKT
  bag: B13 Y16 R16 K18 T17   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       RR | .....
      .YY | .....
     .... | .....
    ..... | .....
    floor: R# (-2)
 P1  score 0
        . | .....
       .. | .....
      ... | .....
     BBBB | .....
    ...TT | .....
    floor: - (0)
```

### 20. Round 1, the expert's move 1 (game 3401, position 0; expert Elo 2460)

- **Expert took** 2 yellow from factory 4 to the 3-tile row. Net's Q for it +nan, 0.0% of visits, raw prior 0.0%.
- **Net wanted** 2 teal from factory 1 to the 4-tile row. Q +0.409, 98.7% of visits, raw prior 80.2%.
- Searched from the children: after the expert's move -0.394, after the net's +0.414, **child-search loss 0.808**.
- Root-edge value loss **nan** (+nan points by the margin head); root value +0.405. The expert won the game 78-60.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: BYKT
  1: BKTT
  2: BYRK
  3: BBRK
  4: BYYT
  center: - [1st]
  bag: B14 Y16 R18 K16 T16   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
```


## 5. The largest round-2 disagreements

### 1. Round 2, the expert's move 4 (game 2865, position 18; expert Elo 2460)

- **Expert took** 2 blue from the center to the 5-tile row. Net's Q for it -0.747, 0.0% of visits, raw prior 2.6%.
- **Net wanted** 2 black from the center to the 5-tile row. Q +0.293, 97.3% of visits, raw prior 51.9%.
- Root-edge value loss **1.040** (+11.8 points by the margin head); root value +0.283. The expert won the game 58-29.

```
round 1  to move: P0  first player: P1  scores: 4-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: BBYYYYRKK
  bag: B10 Y7 R16 K12 T15   lid: B5 Y5 R0 K0 T0
*P0  score 4
        T | ...K.
       BB | .....
      KKK | ...Y.
     .... | ...B.
    ..... | .....
    floor: # (-1)
 P1  score 0
        Y | ..R..
       RR | ..Y..
      ... | ...Y.
     TTTT | .....
    ...KK | .....
    floor: - (0)
```

### 2. Round 2, the expert's move 5 (game 2813, position 20; expert Elo 2033)

- **Expert took** 2 blue from the center to the 1-tile row, completing it, 1 to the floor. Net's Q for it -0.317, 0.0% of visits, raw prior 1.4%.
- **Net wanted** 1 black from factory 3 to the 1-tile row, completing it. Q +0.635, 96.9% of visits, raw prior 7.9%.
- Root-edge value loss **0.952** (+10.2 points by the margin head); root value +0.613. The expert lost the game 40-38.

```
round 1  to move: P1  first player: P1  scores: 3-1
Factories:
  0: -
  1: -
  2: -
  3: BBKT
  4: -
  center: BBYT
  bag: B8 Y15 R12 K11 T14   lid: B3 Y0 R0 K0 T0
 P0  score 3
        . | .Y...
       YY | .B...
      BBB | .....
     KKKK | .....
    .RRRR | .....
    floor: - (0)
*P1  score 1
        . | .Y...
       .T | .B...
      TTT | .....
     KKKK | .....
    .RRRR | .....
    floor: # (-1)
```

### 3. Round 2, the expert's move 5 (game 3587, position 21; expert Elo 2460)

- **Expert took** 2 yellow from the center to the 5-tile row. Net's Q for it -0.633, 0.0% of visits, raw prior 0.8%.
- **Net wanted** 2 yellow from the center straight to the floor. Q +0.297, 59.8% of visits, raw prior 30.8%.
- Root-edge value loss **0.930** (+9.9 points by the margin head); root value +0.287. The expert won the game 59-44.

```
round 1  to move: P0  first player: P1  scores: 2-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: YYK
  bag: B12 Y13 R13 K10 T12   lid: B3 Y2 R1 K3 T3
*P0  score 2
        R | ...K.
       YY | ...R.
      TTT | .....
     .... | .K...
    ..... | .....
    floor: # (-1)
 P1  score 0
        T | .Y...
       .. | .B...
      BBB | .T...
     KKKK | .....
    .RRRR | .....
    floor: B (-1)
```

### 4. Round 2, the expert's move 6 (game 2209, position 21; expert Elo 2460)

- **Expert took** 1 blue from the center to the 5-tile row. Net's Q for it -0.049, 0.0% of visits, raw prior 1.6%.
- **Net wanted** 4 black from the center to the 5-tile row. Q +0.873, 98.8% of visits, raw prior 90.5%.
- Root-edge value loss **0.922** (+13.1 points by the margin head); root value +0.871. The expert won the game 64-41.

```
round 1  to move: P0  first player: P0  scores: 4-3
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: BKKKK
  bag: B15 Y12 R13 K7 T13   lid: B3 Y3 R0 K4 T1
*P0  score 4
        Y | ..R..
       RR | ..Y..
      RRR | ..B..
     TTTT | .....
    ..... | .....
    floor: - (0)
 P1  score 3
        T | ...K.
       YY | T....
      KKK | ...Y.
     .... | .K...
    ....R | .....
    floor: # (-1)
```

### 5. Round 2, the expert's move 3 (game 1200, position 18; expert Elo 2486)

- **Expert took** 2 blue from factory 4 to the 1-tile row, completing it, 1 to the floor. Net's Q for it -0.652, 0.1% of visits, raw prior 5.4%.
- **Net wanted** 3 yellow from the center to the 3-tile row, completing it. Q +0.268, 99.4% of visits, raw prior 74.3%.
- Root-edge value loss **0.921** (+9.6 points by the margin head); root value +0.263. The expert won the game 54-15.

```
round 1  to move: P0  first player: P1  scores: 3-3
Factories:
  0: -
  1: -
  2: -
  3: -
  4: BBRR
  center: BYYYRR
  bag: B11 Y13 R13 K12 T11   lid: B2 Y2 R2 K4 T1
*P0  score 3
        . | .Y...
       BB | T....
      ... | ..B..
     .TTT | .....
    ..... | .....
    floor: - (0)
 P1  score 3
        . | B....
       .. | ...R.
      KKK | ...Y.
     TTTT | .....
    ..... | ..K..
    floor: # (-1)
```

### 6. Round 2, the expert's move 7 (game 2572, position 24; expert Elo 2460)

- **Expert took** 1 yellow from the center to the 5-tile row. Net's Q for it -0.291, 0.0% of visits, raw prior 1.3%.
- **Net wanted** 1 yellow from the center straight to the floor. Q +0.615, 74.8% of visits, raw prior 55.0%.
- Root-edge value loss **0.906** (+11.4 points by the margin head); root value +0.607. The expert won the game 46-35.

```
round 1  to move: P0  first player: P0  scores: 1-2
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: BY
  bag: B10 Y15 R13 K11 T11   lid: B3 Y2 R1 K0 T3
*P0  score 1
        Y | ...K.
       RR | ..Y..
      .KK | .T...
     KKKK | .....
    ..... | .....
    floor: B (-1)
 P1  score 2
        T | ...K.
       .. | ...R.
      TTT | ..B..
     BBBB | .....
    ..RRR | .....
    floor: KT# (-4)
```

### 7. Round 2, the expert's move 2 (game 1450, position 15; expert Elo 2460)

- **Expert took** 2 teal from factory 2 to the 5-tile row. Net's Q for it -0.491, 0.0% of visits, raw prior 0.5%.
- **Net wanted** 2 black from factory 1 to the 5-tile row. Q +0.393, 99.8% of visits, raw prior 90.2%.
- Root-edge value loss **0.884** (+9.7 points by the margin head); root value +0.392. The expert lost the game 41-44.

```
round 1  to move: P0  first player: P0  scores: 1-0
Factories:
  0: YRKT
  1: RRKK
  2: YYTT
  3: YYRT
  4: -
  center: -
  bag: B13 Y11 R13 K13 T10   lid: B0 Y3 R2 K1 T1
*P0  score 1
        . | ...K.
       .. | ....K
      ... | ...Y.
     BBBB | .....
    ..... | .....
    floor: B (-1)
 P1  score 0
        . | ..R..
       .. | T....
      .BB | .....
     TTTT | .....
    ....K | .....
    floor: # (-1)
```

### 8. Round 2, the expert's move 4 (game 235, position 16; expert Elo 2486)

- **Expert took** 1 yellow from factory 0 to the 1-tile row, completing it. Net's Q for it -0.272, 0.1% of visits, raw prior 3.2%.
- **Net wanted** 2 red from factory 0 to the 5-tile row (already 2/5). Q +0.595, 99.0% of visits, raw prior 65.5%.
- Root-edge value loss **0.867** (+9.0 points by the margin head); root value +0.588. The expert won the game 56-44.

```
round 1  to move: P0  first player: P0  scores: 1-2
Factories:
  0: YRRK
  1: -
  2: -
  3: -
  4: -
  center: YYK
  bag: B12 Y13 R13 K10 T12   lid: B0 Y1 R2 K6 T1
*P0  score 1
        . | ....T
       BB | ..Y..
      TTT | .....
     .... | .K...
    ...RR | .....
    floor: - (0)
 P1  score 2
        . | B....
       YY | ...R.
      TTT | .....
     BBBB | .K...
    ..... | .....
    floor: B# (-2)
```

### 9. Round 2, the expert's move 5 (game 3044, position 20; expert Elo 2460)

- **Expert took** 2 blue from the center to the 1-tile row, completing it, 1 to the floor. Net's Q for it -0.361, 0.6% of visits, raw prior 25.0%.
- **Net wanted** 4 red from the center to the 1-tile row, completing it, 3 to the floor. Q +0.470, 98.0% of visits, raw prior 43.5%.
- Root-edge value loss **0.832** (+8.9 points by the margin head); root value +0.459. The expert won the game 28-16.

```
round 1  to move: P0  first player: P1  scores: 2-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: BBRRRR
  bag: B10 Y11 R11 K11 T17   lid: B3 Y1 R0 K3 T0
*P0  score 2
        . | .Y...
       TT | ..Y..
      ... | .....
     KKKK | .....
    RRRRR | .....
    floor: - (0)
 P1  score 0
        K | ....T
       YY | .B...
      YYY | .....
     BBBB | .K...
    ..... | .....
    floor: Y# (-2)
```

### 10. Round 2, the expert's move 3 (game 3226, position 17; expert Elo 2083)

- **Expert took** 2 red from the center to the 3-tile row. Net's Q for it -0.458, 0.0% of visits, raw prior 2.1%.
- **Net wanted** 2 blue from factory 0 to the 3-tile row. Q +0.367, 97.3% of visits, raw prior 12.2%.
- Root-edge value loss **0.825** (+9.6 points by the margin head); root value +0.346. The expert won the game 50-45.

```
round 1  to move: P0  first player: P1  scores: 4-2
Factories:
  0: BBRT
  1: -
  2: YYRT
  3: -
  4: -
  center: BRRT
  bag: B10 Y11 R11 K15 T13   lid: B4 Y1 R0 K0 T2
*P0  score 4
        . | B....
       .. | .B...
      ... | .T...
     YYYY | .....
    RRRRR | .....
    floor: # (-1)
 P1  score 2
        Y | ....T
       .. | ..Y..
      ... | .....
     .... | ...B.
    KKKKK | .....
    floor: - (0)
```

