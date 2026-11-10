# Brick mode scoreboard

| brick size | pass | split | fail | error | with floating bricks |
|---|---|---|---|---|---|
| 6 mm | 30 | 0 | 0 | 0 | 20 of 30 |
| 8 mm | 27 | 3 | 0 | 0 | 17 of 30 |
| 12 mm | 28 | 1 | 1 | 0 | 16 of 30 |
| 16 mm | 29 | 0 | 1 | 0 | 13 of 30 |

`Np` = pieces, `/Ns` = largest stray group in bricks, `Nb` = bricks, `Nf` = floating bricks (resting on nothing, so they cannot be laid from the bottom up).

| shape | 6 mm | 8 mm | 12 mm | 16 mm | fidelity @8mm |
|---|---|---|---|---|---|
| cube | PASS 1p 1752b | PASS 1p 767b | PASS 1p 228b | PASS 1p 119b | 92% |
| pyramid | PASS 1p 721b | PASS 1p 332b | PASS 1p 111b | PASS 1p 49b | 85% |
| cone | PASS 1p 588b | SPLIT 2p/1s 283b | PASS 1p 70b | PASS 1p 54b | 89% |
| capsule | PASS 1p 527b 9f | PASS 1p 212b 4f | PASS 1p 81b 1f | PASS 1p 32b | 88% |
| sphere | PASS 1p 1396b 21f | PASS 1p 631b 17f | PASS 1p 227b 8f | PASS 1p 92b 4f | 91% |
| bowl | PASS 1p 535b 21f | PASS 1p 270b 18f | PASS 1p 108b 7f | PASS 1p 52b 4f | 75% |
| cup | PASS 1p 502b 2f | PASS 1p 256b 3f | PASS 1p 88b 1f | PASS 1p 49b | 69% |
| torus | PASS 1p 521b 4f | PASS 1p 237b | PASS 1p 87b | PASS 1p 39b | 85% |
| arch | PASS 1p 266b 6f | PASS 1p 128b 2f | PASS 1p 51b | PASS 1p 18b | 79% |
| vase | PASS 1p 564b | SPLIT 3p/1s 271b | PASS 1p 99b | PASS 1p 53b | 69% |
| tray | PASS 1p 203b | SPLIT 5p/1s 87b | PASS 1p 33b | PASS 1p 15b | 74% |
| table | PASS 1p 382b 62f | PASS 1p 128b 34f | PASS 1p 93b 12f | PASS 1p 46b 5f | 82% |
| chair | PASS 1p 157b 24f | PASS 1p 99b 9f | PASS 1p 50b 2f | PASS 1p 27b 2f | 66% |
| bridge | PASS 1p 169b 7f | PASS 1p 74b | PASS 1p 34b | PASS 1p 15b | 61% |
| fence | PASS 1p 111b | PASS 1p 47b | FAIL 3p/6s 18b | FAIL 3p/1s 6b 2f | 68% |
| stairs | PASS 1p 416b | PASS 1p 183b | PASS 1p 62b | PASS 1p 29b | 90% |
| letter_L | PASS 1p 144b | PASS 1p 67b | PASS 1p 41b | PASS 1p 11b | 78% |
| letter_H | PASS 1p 249b 3f | PASS 1p 110b 1f | PASS 1p 72b | PASS 1p 28b | 79% |
| tower | PASS 1p 72b | PASS 1p 18b | PASS 1p 23b | PASS 1p 9b | 97% |
| mushroom | PASS 1p 553b 70f | PASS 1p 250b 45f | SPLIT 2p/1s 83b 18f | PASS 1p 44b 7f | 77% |
| t_shape | PASS 1p 126b 9f | PASS 1p 61b 6f | PASS 1p 25b 2f | PASS 1p 14b 1f | 76% |
| tree | PASS 1p 509b 18f | PASS 1p 257b 14f | PASS 1p 88b 5f | PASS 1p 51b 3f | 85% |
| shelf | PASS 1p 350b 41f | PASS 1p 129b 24f | PASS 1p 58b 6f | PASS 1p 28b 5f | 92% |
| blob | PASS 1p 796b 17f | PASS 1p 361b 7f | PASS 1p 128b 2f | PASS 1p 60b 1f | 87% |
| car | PASS 1p 315b 19f | PASS 1p 140b 7f | PASS 1p 47b 3f | PASS 1p 20b | 77% |
| animal | PASS 1p 369b 12f | PASS 1p 189b 8f | PASS 1p 73b 2f | PASS 1p 33b 1f | 70% |
| house | PASS 1p 855b | PASS 1p 362b | PASS 1p 118b | PASS 1p 51b | 89% |
| open_sphere | PASS 1p 1396b 21f | PASS 1p 631b 17f | PASS 1p 227b 8f | PASS 1p 92b 4f | 91% |
| inverted_sphere | PASS 1p 1396b 21f | PASS 1p 631b 17f | PASS 1p 227b 8f | PASS 1p 92b 4f | 91% |
| two_bodies | PASS 2p 125b 5f | PASS 2p 58b | PASS 2p 25b 1f | PASS 2p 17b | 74% |
