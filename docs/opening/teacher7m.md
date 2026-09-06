# Opening atlas: rlx_teacher at 4,096 sims

*Generated from `data/cloud/opening/teacher7m_r01_s4096.npz`: 86,613 positions of the expert games in rounds 1-2 (both seats), 45,876 of them the expert's (the dataset's target seat). Search: no root noise, 4,096 simulations per position, c_puct 1.4, 4 determinizations per refill edge. Labelled in 160 min on cuda.*

Columns: **top-1** = the move the net would play (decisive pick) equals the expert's; **top-1 (visits)** = most-visited move equals the expert's; **top-3** = the expert's move is among the three most visited; **raw prior top-1** = the policy head alone, no search; **expert visit share** = the fraction of the search that went into the expert's move; **loss** = Q(net's move) - Q(expert's move) in the mover's frame, win-probability units on [-1, 1] (0.10 ≈ 5 percentage points of win chance); **pts** = the same gap read from the margin head, in final-score points; **unvisited** = the search never tried the expert's move (the loss is then unknown and excluded).

## 1. Agreement by move

**Round 1** (the target seat, i.e. the expert; the last row is the other seat for contrast)

| slice | n | top-1 | top-1 (visits) | top-3 | raw prior top-1 | expert visit share | mean loss | median loss | p90 loss | loss>0.05 | loss>0.10 | loss>0.20 | mean pts | unvisited |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round 1, move 1 | 4,030 | 21.0% | 22.7% | 52.4% | 28.5% | 22.0% | 0.261 | 0.161 | 0.729 | 66.7% | 59.0% | 44.6% | +4.31 | 2.5% |
| round 1, move 2 | 4,030 | 24.6% | 29.5% | 63.2% | 38.2% | 27.5% | 0.151 | 0.056 | 0.443 | 51.8% | 39.8% | 26.3% | +2.54 | 2.1% |
| round 1, move 3 | 4,030 | 32.9% | 37.9% | 69.7% | 39.4% | 33.9% | 0.108 | 0.031 | 0.323 | 43.8% | 31.8% | 18.2% | +1.81 | 2.7% |
| round 1, move 4 | 4,030 | 43.7% | 46.7% | 78.3% | 44.0% | 43.6% | 0.093 | 0.012 | 0.283 | 38.9% | 28.8% | 16.3% | +1.59 | 1.7% |
| round 1, move 5 | 3,974 | 61.2% | 64.5% | 90.7% | 61.1% | 61.2% | 0.057 | 0.000 | 0.208 | 24.5% | 18.4% | 10.4% | +1.02 | 0.4% |
| round 1, move 6 | 2,706 | 74.8% | 79.7% | 97.3% | 78.6% | 76.7% | 0.026 | 0.000 | 0.071 | 12.2% | 7.9% | 4.3% | +0.51 | 2.9% |
| round 1, move 7 | 629 | 83.5% | 87.1% | 99.2% | 87.0% | 85.3% | 0.017 | 0.000 | 0.036 | 9.0% | 5.1% | 2.7% | +0.33 | 10.0% |
| round 1, move 8 | 49 | 85.7% | 89.8% | 100.0% | 89.8% | 87.7% | 0.016 | 0.000 | 0.024 | 10.0% | 2.5% | 2.5% | +0.25 | 18.4% |
| round 1, moves 1-3 | 12,090 | 26.2% | 30.0% | 61.7% | 35.4% | 27.8% | 0.173 | 0.068 | 0.528 | 54.1% | 43.5% | 29.7% | +2.89 | 2.4% |
| round 1, all | 23,479 | 42.4% | 46.1% | 74.7% | 47.7% | 43.5% | 0.118 | 0.016 | 0.382 | 40.4% | 31.6% | 20.4% | +2.00 | 2.2% |
| round 1, opponent seat | 20,729 | 27.4% | 29.8% | 58.0% | 36.3% | 28.4% | 0.206 | 0.098 | 0.602 | 58.6% | 49.7% | 36.1% | +3.66 | 4.3% |

**Round 2** (the target seat, i.e. the expert; the last row is the other seat for contrast)

| slice | n | top-1 | top-1 (visits) | top-3 | raw prior top-1 | expert visit share | mean loss | median loss | p90 loss | loss>0.05 | loss>0.10 | loss>0.20 | mean pts | unvisited |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round 2, move 1 | 4,030 | 29.3% | 39.9% | 65.9% | 44.1% | 33.7% | 0.077 | 0.019 | 0.235 | 33.0% | 22.3% | 12.5% | +1.42 | 5.4% |
| round 2, move 2 | 4,030 | 26.8% | 35.1% | 64.5% | 38.1% | 29.2% | 0.087 | 0.021 | 0.271 | 36.0% | 24.7% | 14.1% | +1.50 | 4.8% |
| round 2, move 3 | 4,030 | 37.5% | 41.8% | 71.7% | 38.4% | 36.6% | 0.085 | 0.013 | 0.265 | 36.3% | 25.4% | 14.2% | +1.49 | 2.9% |
| round 2, move 4 | 4,030 | 53.6% | 56.1% | 84.0% | 45.1% | 50.9% | 0.068 | 0.000 | 0.230 | 26.4% | 19.3% | 11.7% | +1.23 | 1.2% |
| round 2, move 5 | 3,829 | 70.4% | 72.0% | 94.8% | 66.0% | 68.0% | 0.039 | 0.000 | 0.116 | 16.1% | 11.4% | 6.4% | +0.70 | 2.4% |
| round 2, move 6 | 2,047 | 80.0% | 81.2% | 97.5% | 78.4% | 78.2% | 0.026 | 0.000 | 0.068 | 11.4% | 8.1% | 4.4% | +0.47 | 13.7% |
| round 2, move 7 | 375 | 81.9% | 82.7% | 99.7% | 83.7% | 82.0% | 0.027 | 0.000 | 0.064 | 10.9% | 6.2% | 4.0% | +0.46 | 26.9% |
| round 2, moves 1-3 | 12,090 | 31.2% | 38.9% | 67.4% | 40.2% | 33.2% | 0.083 | 0.019 | 0.259 | 35.1% | 24.2% | 13.6% | +1.47 | 4.4% |
| round 2, all | 22,397 | 47.3% | 52.3% | 78.4% | 49.8% | 47.3% | 0.067 | 0.000 | 0.219 | 27.9% | 19.5% | 11.1% | +1.19 | 4.7% |
| round 2, opponent seat | 20,008 | 35.5% | 40.1% | 70.0% | 45.1% | 35.9% | 0.099 | 0.025 | 0.297 | 40.1% | 28.3% | 16.1% | +1.99 | 4.9% |

**Round 1 by the expert's Elo** (the source ladder's raw scale; the pipeline's floor is 1950)

| slice | n | top-1 | top-1 (visits) | top-3 | raw prior top-1 | expert visit share | mean loss | median loss | p90 loss | loss>0.05 | loss>0.10 | loss>0.20 | mean pts | unvisited |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| expert Elo < 2200 | 2,067 | 40.3% | 43.3% | 71.3% | 45.8% | 41.0% | 0.134 | 0.026 | 0.415 | 44.5% | 36.0% | 23.9% | +2.24 | 2.1% |
| 2200-2350 | 1,066 | 45.5% | 50.1% | 79.3% | 52.4% | 46.2% | 0.102 | 0.009 | 0.335 | 36.3% | 28.4% | 17.2% | +1.79 | 1.4% |
| ≥ 2350 | 20,346 | 42.4% | 46.2% | 74.8% | 47.6% | 43.6% | 0.118 | 0.016 | 0.380 | 40.2% | 31.3% | 20.3% | +1.99 | 2.3% |

## 2. How much the net thinks the expert lost

**Round 1, expert's decisions**

| quantile | value loss | points |
|---|---|---|
| p50 | 0.016 | +0.41 |
| p75 | 0.152 | +2.66 |
| p90 | 0.382 | +6.31 |
| p95 | 0.580 | +9.42 |
| p99 | 0.937 | +15.19 |
| mean | 0.118 | +2.00 |

Disagreements: 57.3% of decisions. Among them the mean value loss is 0.207 (+3.50 points); 15.3% are within 0.02 of the net's own pick (a coin flip to the net), 55.1% are what the net calls a clear mistake (> 0.10), 35.7% a blunder (> 0.20).

**Rounds 1-2, expert's decisions**

| quantile | value loss | points |
|---|---|---|
| p50 | 0.006 | +0.23 |
| p75 | 0.106 | +2.01 |
| p90 | 0.307 | +5.01 |
| p95 | 0.481 | +7.69 |
| p99 | 0.866 | +13.85 |
| mean | 0.094 | +1.61 |

Disagreements: 55.0% of decisions. Among them the mean value loss is 0.170 (+2.93 points); 20.9% are within 0.02 of the net's own pick (a coin flip to the net), 46.8% are what the net calls a clear mistake (> 0.10), 29.0% a blunder (> 0.20).

## 3. What the disagreements look like (round 1)

Disagreements with value loss > 0.05 (target seat): 9,266. Coarse patterns (a disagreement can count in two rows):

| pattern | count | share |
|---|---|---|
| different colour | 6925 | 74.7% |
| center vs factory: expert center | 2113 | 22.8% |
| same colour, other row (expert higher row) | 906 | 9.8% |
| center vs factory: net center | 821 | 8.9% |
| same colour, other row (expert lower row) | 692 | 7.5% |
| same colour and row, other source | 398 | 4.3% |
| expert builds, net floors | 188 | 2.0% |
| expert floors, net builds | 157 | 1.7% |

Finest patterns, top 12:

| pattern | count |
|---|---|
| different colour; same row | 612 |
| different colour; different factory; row 3 vs row 5 (expert higher) | 536 |
| same colour; row 4 vs row 5 (expert higher) | 364 |
| different colour; different factory; same row | 332 |
| different colour; different factory; row 4 vs row 5 (expert higher) | 313 |
| different colour; different factory; row 3 vs row 4 (expert higher) | 279 |
| different colour; expert center, net factory; row 3 vs row 5 (expert higher) | 203 |
| different colour; expert center, net factory; same row | 199 |
| different colour; expert center, net factory; row 1 vs row 5 (expert higher) | 181 |
| different colour; expert center, net factory; row 1 vs row 4 (expert higher) | 169 |
| different colour; row 3 vs row 5 (expert higher) | 169 |
| same colour; expert center, net factory; same row | 168 |

## 4. The 20 largest round-1 disagreements

Boards are printed before the move, from the engine's text renderer: each player's five pattern lines on the left (`.` empty slot), the wall on the right (`.` empty), colours B=blue Y=yellow R=red K=black T=teal, `#` on the floor = the first-player marker; `*` marks the player to move.

### 1. Round 1, the expert's move 4 (game 1830, position 7; expert Elo 2460)

- **Expert took** 2 red from the center to the 2-tile row, completing it. Net's Q for it -0.536, 0.4% of visits, raw prior 29.8%.
- **Net wanted** 3 black from the center to the 5-tile row. Q +0.873, 98.7% of visits, raw prior 18.4%.
- Root-edge value loss **1.408** (+24.8 points by the margin head); root value +0.857. The expert won the game 57-46.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: YYYYRRKKKT
  bag: B16 Y14 R17 K16 T17   lid: B0 Y0 R0 K0 T0
*P0  score 0
        K | .....
       .. | .....
      BBB | .....
     .... | .....
    ..... | .....
    floor: # (-1)
 P1  score 0
        R | .....
       YY | .....
      ..B | .....
     ..TT | .....
    ..... | .....
    floor: - (0)
```

### 2. Round 1, the expert's move 1 (game 1678, position 1; expert Elo 2460)

- **Expert took** 1 yellow from the center to the 3-tile row (takes the first-player marker). Net's Q for it -0.848, 0.0% of visits, raw prior 1.0%.
- **Net wanted** 2 black from factory 1 to the 5-tile row. Q +0.490, 97.9% of visits, raw prior 19.7%.
- Root-edge value loss **1.338** (+23.5 points by the margin head); root value +0.475. The expert won the game 22-11.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: YYRT
  1: BRKK
  2: -
  3: BBYY
  4: YYRT
  center: YK [1st]
  bag: B15 Y13 R17 K17 T18   lid: B0 Y0 R0 K0 T0
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
      .BB | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 3. Round 1, the expert's move 3 (game 410, position 4; expert Elo 2486)

- **Expert took** 2 blue from factory 1 to the 4-tile row (already 3/4), completing it, 1 to the floor. Net's Q for it -0.597, 0.1% of visits, raw prior 9.9%.
- **Net wanted** 3 black from factory 4 to the 5-tile row. Q +0.730, 98.8% of visits, raw prior 38.9%.
- Root-edge value loss **1.327** (+21.7 points by the margin head); root value +0.718. The expert lost the game 39-41.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: -
  1: BBYT
  2: -
  3: -
  4: YKKK
  center: BK
  bag: B14 Y13 R18 K16 T19   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       RR | .....
      ... | .....
     .BBB | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       YY | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: # (-1)
```

### 4. Round 1, the expert's move 1 (game 1792, position 1; expert Elo 2025)

- **Expert took** 1 yellow from the center to the 1-tile row, completing it (takes the first-player marker). Net's Q for it -0.742, 0.1% of visits, raw prior 5.4%.
- **Net wanted** 2 black from factory 1 to the 5-tile row. Q +0.560, 97.7% of visits, raw prior 6.5%.
- Root-edge value loss **1.301** (+22.3 points by the margin head); root value +0.543. The expert lost the game 59-61.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: BYKT
  1: RRKK
  2: BBKT
  3: YRRT
  4: -
  center: YT [1st]
  bag: B15 Y17 R16 K16 T16   lid: B0 Y0 R0 K0 T0
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
      .BB | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 5. Round 1, the expert's move 4 (game 564, position 6; expert Elo 1971)

- **Expert took** 2 blue from factory 4 straight to the floor. Net's Q for it -0.656, 0.2% of visits, raw prior 15.5%.
- **Net wanted** 3 red from the center to the 5-tile row. Q +0.621, 98.8% of visits, raw prior 26.8%.
- Root-edge value loss **1.277** (+19.3 points by the margin head); root value +0.606. The expert lost the game 54-44.

```
round 0  to move: P1  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: BBKT
  center: YRRR
  bag: B13 Y17 R16 K15 T19   lid: B0 Y0 R0 K0 T0
 P0  score 0
        . | .....
       .. | .....
      .BB | .....
     .... | .....
    .KKKK | .....
    floor: - (0)
*P1  score 0
        R | .....
       YY | .....
      BBB | .....
     .... | .....
    ..... | .....
    floor: # (-1)
```

### 6. Round 1, the expert's move 2 (game 526, position 2; expert Elo 2486)

- **Expert took** 1 blue from the center to the 3-tile row (already 2/3), completing it (takes the first-player marker). Net's Q for it -0.628, 1.4% of visits, raw prior 80.9%.
- **Net wanted** 2 black from factory 3 to the 5-tile row. Q +0.641, 98.3% of visits, raw prior 5.8%.
- Root-edge value loss **1.270** (+20.6 points by the margin head); root value +0.622. The expert won the game 58-20.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: -
  1: -
  2: YRKT
  3: YKKT
  4: BYKT
  center: BYR [1st]
  bag: B16 Y13 R18 K16 T17   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      .BB | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 7. Round 1, the expert's move 2 (game 482, position 2; expert Elo 1995)

- **Expert took** 1 black from the center to the 1-tile row, completing it (takes the first-player marker). Net's Q for it -0.788, 0.0% of visits, raw prior 1.8%.
- **Net wanted** 2 black from factory 4 to the 5-tile row. Q +0.479, 97.5% of visits, raw prior 9.8%.
- Root-edge value loss **1.267** (+21.5 points by the margin head); root value +0.451. The expert lost the game 57-29.

```
round 0  to move: P1  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: YYRT
  3: YRRT
  4: KKTT
  center: RKT [1st]
  bag: B18 Y14 R16 K17 T15   lid: B0 Y0 R0 K0 T0
 P0  score 0
        . | .....
       .. | .....
      .BB | .....
     .... | .....
    ..... | .....
    floor: - (0)
*P1  score 0
        . | .....
       .. | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 8. Round 1, the expert's move 2 (game 878, position 2; expert Elo 2460)

- **Expert took** 1 black from the center to the 1-tile row, completing it (takes the first-player marker). Net's Q for it -0.812, 0.0% of visits, raw prior 1.9%.
- **Net wanted** 2 black from factory 0 to the 5-tile row. Q +0.454, 97.5% of visits, raw prior 22.7%.
- Root-edge value loss **1.266** (+22.1 points by the margin head); root value +0.429. The expert won the game 47-27.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: BYKK
  1: -
  2: BYYT
  3: -
  4: RRTT
  center: RKT [1st]
  bag: B16 Y14 R17 K17 T16   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      .BB | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 9. Round 1, the expert's move 2 (game 1148, position 3; expert Elo 2460)

- **Expert took** 2 yellow from the center to the 3-tile row (takes the first-player marker). Net's Q for it -0.722, 0.0% of visits, raw prior 3.1%.
- **Net wanted** 2 black from factory 0 to the 5-tile row. Q +0.543, 94.1% of visits, raw prior 20.4%.
- Root-edge value loss **1.266** (+19.8 points by the margin head); root value +0.528. The expert won the game 58-41.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: YRKK
  1: -
  2: YYKT
  3: -
  4: -
  center: YYRRKT [1st]
  bag: B16 Y15 R17 K16 T16   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       BB | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      ... | .....
     ..BB | .....
    ...TT | .....
    floor: - (0)
```

### 10. Round 1, the expert's move 2 (game 2568, position 2; expert Elo 2007)

- **Expert took** 2 yellow from the center to the 3-tile row (takes the first-player marker). Net's Q for it -0.889, 0.0% of visits, raw prior 0.7%.
- **Net wanted** 2 black from factory 4 to the 5-tile row. Q +0.370, 41.3% of visits, raw prior 13.9%.
- Root-edge value loss **1.259** (+22.5 points by the margin head); root value +0.356. The expert lost the game 50-27.

```
round 0  to move: P1  first player: P1  scores: 0-0
Factories:
  0: -
  1: BYTT
  2: -
  3: YKTT
  4: YRKK
  center: YYKT [1st]
  bag: B17 Y15 R17 K16 T15   lid: B0 Y0 R0 K0 T0
 P0  score 0
        . | .....
       .. | .....
      .BB | .....
     .... | .....
    ..... | .....
    floor: - (0)
*P1  score 0
        . | .....
       RR | .....
      ... | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 11. Round 1, the expert's move 4 (game 493, position 6; expert Elo 2033)

- **Expert took** 3 black from the center to the 2-tile row, completing it, 1 to the floor. Net's Q for it -0.587, 0.3% of visits, raw prior 20.2%.
- **Net wanted** 3 black from the center to the 5-tile row. Q +0.667, 99.3% of visits, raw prior 51.3%.
- Root-edge value loss **1.254** (+20.3 points by the margin head); root value +0.658. The expert won the game 45-51.

```
round 0  to move: P1  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: BYRT
  center: RKKK
  bag: B16 Y13 R17 K17 T17   lid: B0 Y0 R0 K0 T0
 P0  score 0
        R | .....
       .. | .....
      YYY | .....
     ..TT | .....
    ..... | .....
    floor: # (-1)
*P1  score 0
        . | .....
       .. | .....
      YYY | .....
     .BBB | .....
    ..... | .....
    floor: - (0)
```

### 12. Round 1, the expert's move 2 (game 2399, position 2; expert Elo 2460)

- **Expert took** 1 red from the center to the 1-tile row, completing it (takes the first-player marker). Net's Q for it -0.567, 0.3% of visits, raw prior 17.6%.
- **Net wanted** 2 black from factory 1 to the 5-tile row. Q +0.684, 98.8% of visits, raw prior 40.5%.
- Root-edge value loss **1.251** (+20.4 points by the margin head); root value +0.673. The expert won the game 68-31.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: YRKT
  1: YKKT
  2: -
  3: BYRK
  4: -
  center: YR [1st]
  bag: B16 Y16 R17 K16 T15   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      ... | .....
     .TTT | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      BBB | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 13. Round 1, the expert's move 1 (game 2780, position 1; expert Elo 2486)

- **Expert took** 3 blue from factory 1 to the 3-tile row, completing it. Net's Q for it -0.462, 0.5% of visits, raw prior 27.7%.
- **Net wanted** 3 black from factory 4 to the 5-tile row. Q +0.787, 97.9% of visits, raw prior 30.9%.
- Root-edge value loss **1.249** (+20.6 points by the margin head); root value +0.774. The expert won the game 35-11.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: -
  1: BBBT
  2: YYRR
  3: YYRT
  4: RKKK
  center: K [1st]
  bag: B17 Y13 R16 K16 T18   lid: B0 Y0 R0 K0 T0
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
      YYY | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 14. Round 1, the expert's move 3 (game 482, position 4; expert Elo 1995)

- **Expert took** 1 teal from the center to the 5-tile row. Net's Q for it -0.880, 0.0% of visits, raw prior 1.4%.
- **Net wanted** 2 black from the center to the 5-tile row. Q +0.348, 98.2% of visits, raw prior 28.8%.
- Root-edge value loss **1.228** (+23.0 points by the margin head); root value +0.328. The expert lost the game 57-29.

```
round 0  to move: P1  first player: P1  scores: 0-0
Factories:
  0: -
  1: -
  2: YYRT
  3: YRRT
  4: -
  center: RKKT
  bag: B18 Y14 R16 K17 T15   lid: B0 Y0 R0 K0 T0
 P0  score 0
        . | .....
       .. | .....
      .BB | .....
     ..TT | .....
    ..... | .....
    floor: - (0)
*P1  score 0
        K | .....
       .. | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: # (-1)
```

### 15. Round 1, the expert's move 1 (game 129, position 1; expert Elo 2486)

- **Expert took** 2 yellow from the center to the 2-tile row, completing it (takes the first-player marker). Net's Q for it -0.796, 1.1% of visits, raw prior 63.5%.
- **Net wanted** 2 black from factory 1 to the 5-tile row. Q +0.415, 98.3% of visits, raw prior 13.5%.
- Root-edge value loss **1.211** (+21.1 points by the margin head); root value +0.398. The expert won the game 56-29.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: BRKT
  1: BRKK
  2: YRKT
  3: YRKT
  4: -
  center: YY [1st]
  bag: B16 Y16 R16 K15 T17   lid: B0 Y0 R0 K0 T0
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
      .BB | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 16. Round 1, the expert's move 1 (game 1173, position 0; expert Elo 2460)

- **Expert took** 4 red from factory 4 to the 4-tile row, completing it. Net's Q for it -0.611, 0.0% of visits, raw prior 1.2%.
- **Net wanted** 3 black from factory 0 to the 5-tile row. Q +0.597, 97.0% of visits, raw prior 5.9%.
- Root-edge value loss **1.209** (+18.8 points by the margin head); root value +0.577. The expert won the game 75-47.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: KKKT
  1: YKTT
  2: BBYT
  3: YYKT
  4: RRRR
  center: - [1st]
  bag: B18 Y16 R16 K15 T15   lid: B0 Y0 R0 K0 T0
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

### 17. Round 1, the expert's move 2 (game 2074, position 2; expert Elo 2460)

- **Expert took** 2 yellow from the center to the 2-tile row, completing it (takes the first-player marker). Net's Q for it -0.720, 0.5% of visits, raw prior 27.4%.
- **Net wanted** 2 black from factory 0 to the 5-tile row. Q +0.484, 97.8% of visits, raw prior 15.3%.
- Root-edge value loss **1.204** (+20.7 points by the margin head); root value +0.466. The expert won the game 73-43.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: YKKT
  1: BYYK
  2: BRRT
  3: -
  4: -
  center: BYY [1st]
  bag: B15 Y12 R18 K17 T18   lid: B0 Y0 R0 K0 T0
*P0  score 0
        . | .....
       .. | .....
      YYY | .....
     .... | .....
    ..... | .....
    floor: - (0)
 P1  score 0
        . | .....
       .. | .....
      .BB | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 18. Round 1, the expert's move 1 (game 2315, position 1; expert Elo 2460)

- **Expert took** 3 yellow from factory 3 to the 3-tile row, completing it. Net's Q for it -0.832, 0.4% of visits, raw prior 22.0%.
- **Net wanted** 2 black from factory 1 to the 5-tile row. Q +0.369, 97.8% of visits, raw prior 10.1%.
- Root-edge value loss **1.201** (+21.8 points by the margin head); root value +0.351. The expert won the game 56-36.

```
round 0  to move: P0  first player: P1  scores: 0-0
Factories:
  0: YRRT
  1: BBKK
  2: -
  3: YYYK
  4: BYRT
  center: T [1st]
  bag: B14 Y15 R17 K17 T17   lid: B0 Y0 R0 K0 T0
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
      BBB | .....
     .... | .....
    ..... | .....
    floor: - (0)
```

### 19. Round 1, the expert's move 4 (game 616, position 6; expert Elo 2158)

- **Expert took** 2 yellow from the center to the 4-tile row. Net's Q for it -0.931, 0.0% of visits, raw prior 1.6%.
- **Net wanted** 1 black from factory 2 to the 4-tile row. Q +0.267, 96.0% of visits, raw prior 6.2%.
- Root-edge value loss **1.197** (+19.3 points by the margin head); root value +0.246. The expert lost the game 49-8.

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

### 20. Round 1, the expert's move 1 (game 1307, position 0; expert Elo 2460)

- **Expert took** 3 yellow from factory 4 to the 3-tile row, completing it. Net's Q for it -0.663, 0.2% of visits, raw prior 9.8%.
- **Net wanted** 2 teal from factory 2 to the 4-tile row. Q +0.528, 40.1% of visits, raw prior 3.9%.
- Root-edge value loss **1.191** (+19.0 points by the margin head); root value +0.501. The expert won the game 68-49.

```
round 0  to move: P0  first player: P0  scores: 0-0
Factories:
  0: BBYK
  1: BYYT
  2: BKTT
  3: YKTT
  4: BYYY
  center: - [1st]
  bag: B15 Y13 R20 K17 T15   lid: B0 Y0 R0 K0 T0
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

### 1. Round 2, the expert's move 5 (game 1844, position 23; expert Elo 2460)

- **Expert took** 2 red from the center to the 5-tile row. Net's Q for it -0.812, 0.0% of visits, raw prior 3.3%.
- **Net wanted** 4 black from the center to the 5-tile row. Q +0.875, 99.3% of visits, raw prior 63.6%.
- Root-edge value loss **1.687** (+29.7 points by the margin head); root value +0.868. The expert won the game 46-25.

```
round 1  to move: P0  first player: P0  scores: 1-3
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: BYRRKKKK
  bag: B10 Y13 R13 K10 T14   lid: B2 Y1 R1 K4 T2
*P0  score 1
        . | ...K.
       BB | ..Y..
      YYY | .T...
     BBBB | .....
    ..... | .....
    floor: - (0)
 P1  score 3
        R | .Y...
       .. | .....
      RRR | ..B..
     .TTT | .....
    ..... | ..K..
    floor: # (-1)
```

### 2. Round 2, the expert's move 3 (game 493, position 16; expert Elo 2033)

- **Expert took** 2 red from factory 1 to the 3-tile row. Net's Q for it -0.851, 0.0% of visits, raw prior 1.9%.
- **Net wanted** 2 black from factory 3 to the 5-tile row. Q +0.816, 98.5% of visits, raw prior 21.0%.
- Root-edge value loss **1.667** (+29.2 points by the margin head); root value +0.795. The expert won the game 45-51.

```
round 1  to move: P1  first player: P0  scores: 0-0
Factories:
  0: -
  1: RRKT
  2: -
  3: BRKK
  4: -
  center: YR
  bag: B14 Y12 R11 K11 T12   lid: B1 Y5 R1 K2 T0
 P0  score 0
        . | ..R..
       .. | .....
      TTT | ...Y.
     TTTT | .....
    ..KKK | .....
    floor: # (-1)
*P1  score 0
        . | ..R..
       RR | ....K
      ... | ...Y.
     BBBB | .....
    ..... | .....
    floor: - (0)
```

### 3. Round 2, the expert's move 1 (game 3306, position 11; expert Elo 2460)

- **Expert took** 2 yellow from factory 3 to the 3-tile row. Net's Q for it -0.773, 0.0% of visits, raw prior 3.1%.
- **Net wanted** 2 black from factory 0 to the 5-tile row. Q +0.665, 99.3% of visits, raw prior 71.1%.
- Root-edge value loss **1.438** (+22.4 points by the margin head); root value +0.661. The expert won the game 70-44.

```
round 1  to move: P0  first player: P0  scores: 3-3
Factories:
  0: BKKT
  1: BYRT
  2: YRRK
  3: BYYT
  4: YRKT
  center: - [1st]
  bag: B13 Y12 R11 K10 T14   lid: B2 Y2 R3 K4 T0
*P0  score 3
        . | ...K.
       .. | ...R.
      ... | ..B..
     ..TT | .....
    ..... | .....
    floor: - (0)
 P1  score 3
        . | B....
       .. | ..Y..
      ... | ....R
     .... | .....
    ..... | ..K..
    floor: - (0)
```

### 4. Round 2, the expert's move 2 (game 192, position 16; expert Elo 2173)

- **Expert took** 3 yellow from the center to the 4-tile row. Net's Q for it -0.930, 0.0% of visits, raw prior 1.1%.
- **Net wanted** 1 teal from factory 3 to the 3-tile row (already 2/3), completing it. Q +0.483, 26.4% of visits, raw prior 30.9%.
- Root-edge value loss **1.413** (+22.3 points by the margin head); root value +0.480. The expert won the game 30-33.

```
round 1  to move: P1  first player: P0  scores: 0-1
Factories:
  0: KKTT
  1: -
  2: -
  3: BYKT
  4: BYKT
  center: YYY
  bag: B13 Y9 R13 K13 T12   lid: B0 Y4 R1 K2 T1
 P0  score 0
        R | ...K.
       .. | ...R.
      ... | .....
     BBBB | .....
    ....T | .....
    floor: R (-1)
*P1  score 1
        . | B....
       .. | ..Y..
      .TT | ...Y.
     .... | .....
    ..RRR | .....
    floor: # (-1)
```

### 5. Round 2, the expert's move 4 (game 1397, position 17; expert Elo 2029)

- **Expert took** 1 blue from factory 4 straight to the floor. Net's Q for it -0.456, 0.0% of visits, raw prior 4.2%.
- **Net wanted** 1 teal from factory 4 to the 4-tile row (already 1/4). Q +0.814, 98.3% of visits, raw prior 30.3%.
- Root-edge value loss **1.269** (+19.8 points by the margin head); root value +0.800. The expert won the game 41-38.

```
round 1  to move: P0  first player: P0  scores: 1-0
Factories:
  0: -
  1: -
  2: -
  3: -
  4: BKKT
  center: YRKK
  bag: B12 Y13 R11 K9 T15   lid: B0 Y4 R3 K3 T0
*P0  score 1
        . | B....
       RR | ..Y..
      BBB | ...Y.
     ...T | .....
    ..... | .....
    floor: R# (-2)
 P1  score 0
        . | ..R..
       KK | ...R.
      ... | K....
     .BBB | .....
    ..TTT | .....
    floor: K (-1)
```

### 6. Round 2, the expert's move 3 (game 82, position 15; expert Elo 2486)

- **Expert took** 1 black from factory 4 straight to the floor. Net's Q for it -0.418, 0.0% of visits, raw prior 2.4%.
- **Net wanted** 1 black from factory 4 to the 5-tile row. Q +0.846, 97.9% of visits, raw prior 3.2%.
- Root-edge value loss **1.264** (+21.0 points by the margin head); root value +0.821. The expert won the game 53-37.

```
round 1  to move: P0  first player: P0  scores: 1-4
Factories:
  0: -
  1: YKTT
  2: -
  3: -
  4: BRRK
  center: BRRTT
  bag: B14 Y10 R10 K12 T14   lid: B3 Y2 R1 K1 T1
*P0  score 1
        . | ...K.
       RR | T....
      YYY | .....
     .... | ...B.
    ..... | .....
    floor: - (0)
 P1  score 4
        . | .Y...
       YY | ...R.
      ... | ...Y.
     ..RR | .....
    .KKKK | .....
    floor: # (-1)
```

### 7. Round 2, the expert's move 1 (game 2068, position 11; expert Elo 2460)

- **Expert took** 1 black from the center to the 5-tile row (takes the first-player marker). Net's Q for it -0.472, 0.3% of visits, raw prior 16.4%.
- **Net wanted** 2 black from factory 1 to the 5-tile row. Q +0.789, 88.7% of visits, raw prior 20.7%.
- Root-edge value loss **1.261** (+19.7 points by the margin head); root value +0.775. The expert won the game 69-54.

```
round 1  to move: P0  first player: P1  scores: 3-2
Factories:
  0: -
  1: RKKT
  2: BYKT
  3: BBYT
  4: BYRK
  center: K [1st]
  bag: B15 Y13 R12 K12 T8   lid: B1 Y0 R4 K1 T5
*P0  score 3
        . | .Y...
       .. | ...R.
      ... | .T...
     .... | ..T..
    ..... | .....
    floor: - (0)
 P1  score 2
        . | ...K.
       .. | ....K
      YYY | ....R
     ..TT | .....
    ..... | .....
    floor: - (0)
```

### 8. Round 2, the expert's move 5 (game 3566, position 17; expert Elo 2460)

- **Expert took** 2 yellow from the center to the 5-tile row. Net's Q for it -0.812, 0.0% of visits, raw prior 2.2%.
- **Net wanted** 2 black from the center to the 5-tile row. Q +0.448, 99.7% of visits, raw prior 83.2%.
- Root-edge value loss **1.260** (+17.1 points by the margin head); root value +0.445. The expert won the game 62-39.

```
round 1  to move: P0  first player: P0  scores: 6-5
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: YYKK
  bag: B14 Y10 R11 K14 T11   lid: B0 Y5 R3 K2 T2
*P0  score 6
        T | ...K.
       RR | T....
      RRR | K....
     .BBB | R....
    ..... | .....
    floor: T# (-2)
 P1  score 5
        Y | B....
       BB | T....
      ... | ...Y.
     .... | ....Y
    ..TTT | .....
    floor: - (0)
```

### 9. Round 2, the expert's move 6 (game 1734, position 23; expert Elo 2460)

- **Expert took** 2 teal from the center straight to the floor. Net's Q for it -0.375, 0.0% of visits, raw prior 2.9%.
- **Net wanted** 2 teal from the center to the 1-tile row, completing it, 1 to the floor. Q +0.882, 77.7% of visits, raw prior 37.3%.
- Root-edge value loss **1.258** (+20.7 points by the margin head); root value +0.869. The expert won the game 68-43.

```
round 1  to move: P0  first player: P0  scores: 2-2
Factories:
  0: -
  1: -
  2: -
  3: -
  4: -
  center: BBYRTT
  bag: B10 Y12 R12 K14 T12   lid: B3 Y3 R1 K0 T0
*P0  score 2
        . | ..R..
       YY | .B...
      BBB | ...Y.
     TTTT | .....
    ...KK | .....
    floor: # (-1)
 P1  score 2
        Y | ..R..
       .. | ...R.
      RRR | ..B..
     ..TT | .....
    .KKKK | .....
    floor: - (0)
```

### 10. Round 2, the expert's move 1 (game 3252, position 10; expert Elo 2460)

- **Expert took** 2 blue from factory 0 to the 2-tile row, completing it. Net's Q for it -0.708, 0.0% of visits, raw prior 1.8%.
- **Net wanted** 3 black from factory 2 to the 5-tile row. Q +0.530, 97.1% of visits, raw prior 14.1%.
- Root-edge value loss **1.238** (+19.1 points by the margin head); root value +0.514. The expert won the game 58-53.

```
round 1  to move: P0  first player: P0  scores: 5-2
Factories:
  0: BBKK
  1: YYKT
  2: RKKK
  3: YYRT
  4: YYKT
  center: - [1st]
  bag: B13 Y8 R13 K11 T15   lid: B3 Y4 R2 K0 T0
*P0  score 5
        . | ..R..
       .. | ...R.
      ... | ...Y.
     .... | ...B.
    ..... | .....
    floor: - (0)
 P1  score 2
        . | ..R..
       .. | ..Y..
      ..B | .....
     ..TT | .....
    ...KK | .....
    floor: - (0)
```

