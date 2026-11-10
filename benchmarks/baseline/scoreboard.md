# Brick mode scoreboard

| brick size | pass | split | fail | error | with floating bricks |
|---|---|---|---|---|---|
| 6 mm | 28 | 0 | 2 | 0 | 22 of 30 |
| 8 mm | 24 | 2 | 4 | 0 | 20 of 30 |
| 12 mm | 22 | 4 | 4 | 0 | 17 of 30 |
| 16 mm | 24 | 4 | 2 | 0 | 14 of 30 |

`Np` = pieces, `/Ns` = largest stray group in bricks, `Nb` = bricks, `Nf` = floating bricks (resting on nothing, so they cannot be laid from the bottom up).

| shape | 6 mm | 8 mm | 12 mm | 16 mm | fidelity @8mm |
|---|---|---|---|---|---|
| cube | PASS 1p 1939b | PASS 1p 851b | PASS 1p 268b | PASS 1p 117b | 92% |
| pyramid | FAIL 20p/60s 725b | FAIL 18p/31s 326b | FAIL 10p/12s 114b | FAIL 5p/4s 46b | 85% |
| cone | FAIL 14p/19s 589b | FAIL 7p/11s 285b | PASS 1p 75b | SPLIT 5p/2s 50b | 89% |
| capsule | PASS 1p 536b 15f | PASS 1p 228b 7f | PASS 1p 87b 2f | PASS 1p 38b 1f | 88% |
| sphere | PASS 1p 1425b 68f | PASS 1p 652b 33f | PASS 1p 222b 17f | PASS 1p 102b 6f | 91% |
| bowl | PASS 1p 575b 66f | PASS 1p 290b 40f | PASS 1p 107b 12f | PASS 1p 53b 6f | 75% |
| cup | PASS 1p 553b 8f | FAIL 6p/7s 262b 8f | SPLIT 3p/1s 90b 2f | PASS 1p 51b | 69% |
| torus | PASS 1p 545b 17f | PASS 1p 248b 5f | PASS 1p 94b | PASS 1p 42b | 85% |
| arch | PASS 1p 311b 8f | PASS 1p 154b 3f | PASS 1p 53b | PASS 1p 28b | 79% |
| vase | PASS 1p 589b 21f | SPLIT 4p/1s 271b 5f | SPLIT 2p/1s 103b 1f | PASS 1p 53b | 69% |
| tray | PASS 1p 218b | SPLIT 9p/1s 96b | SPLIT 3p/1s 34b | PASS 1p 15b | 74% |
| table | PASS 1p 387b 60f | PASS 1p 161b 34f | SPLIT 5p/1s 94b 17f | PASS 1p 46b 5f | 82% |
| chair | PASS 1p 220b 21f | PASS 1p 105b 12f | PASS 1p 53b 2f | PASS 1p 27b 2f | 66% |
| bridge | PASS 1p 166b 8f | PASS 1p 84b | PASS 1p 40b | PASS 1p 19b | 61% |
| fence | PASS 1p 116b | PASS 1p 47b | FAIL 3p/6s 18b | FAIL 3p/1s 6b 2f | 68% |
| stairs | PASS 1p 444b | PASS 1p 187b | PASS 1p 74b | PASS 1p 38b | 90% |
| letter_L | PASS 1p 173b | PASS 1p 90b | PASS 1p 46b | PASS 1p 24b | 78% |
| letter_H | PASS 1p 284b 6f | PASS 1p 155b 2f | PASS 1p 77b | PASS 1p 38b | 79% |
| tower | PASS 1p 72b | PASS 1p 44b | PASS 1p 23b | PASS 1p 9b | 97% |
| mushroom | PASS 1p 582b 85f | FAIL 5p/6s 252b 49f | FAIL 7p/8s 86b 21f | SPLIT 3p/1s 50b 12f | 77% |
| t_shape | PASS 1p 145b 15f | PASS 1p 62b 6f | PASS 1p 29b 2f | SPLIT 2p/1s 19b 2f | 76% |
| tree | PASS 1p 492b 24f | PASS 1p 272b 25f | PASS 1p 91b 10f | PASS 1p 53b 6f | 85% |
| shelf | PASS 1p 355b 48f | PASS 1p 196b 38f | PASS 1p 76b 6f | PASS 1p 33b 6f | 92% |
| blob | PASS 1p 813b 37f | PASS 1p 376b 16f | PASS 1p 130b 5f | PASS 1p 67b 3f | 87% |
| car | PASS 1p 322b 26f | PASS 1p 157b 10f | PASS 1p 51b 4f | PASS 1p 26b | 77% |
| animal | PASS 1p 373b 24f | PASS 1p 195b 19f | PASS 1p 77b 7f | SPLIT 2p/2s 37b 2f | 70% |
| house | PASS 1p 874b 1f | PASS 1p 383b | PASS 1p 132b | PASS 1p 59b | 89% |
| open_sphere | PASS 1p 1425b 68f | PASS 1p 652b 33f | PASS 1p 222b 17f | PASS 1p 102b 6f | 91% |
| inverted_sphere | PASS 1p 1425b 68f | PASS 1p 652b 33f | PASS 1p 222b 17f | PASS 1p 102b 6f | 91% |
| two_bodies | PASS 2p 127b 9f | PASS 2p 69b 2f | FAIL 4p/14s 32b 2f | PASS 2p 17b | 74% |
