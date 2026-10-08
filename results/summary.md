# Results (300 frames x 2 seeds per scenario, 640x480)

| Scenario    |   Detection rate % |   False-positive frames % |   Mean centroid err (px) |   Smoothed err (px) |   Yaw cmd acc % |   Heave cmd acc % |   Surge cmd acc % |   Full cmd acc % |   Mask IoU |   Spurious blobs/frame |   Cmd changes /100 fr |   Occluded frames |    FPS |
|:------------|-------------------:|--------------------------:|-------------------------:|--------------------:|----------------:|------------------:|------------------:|-----------------:|-----------:|-----------------------:|----------------------:|------------------:|-------:|
| clear       |                100 |                      0    |                     0.96 |                5.6  |           98.33 |             97.33 |               100 |            95.67 |       0.94 |                   0    |                  4.35 |                 0 | 257.67 |
| turbid      |                100 |                      0    |                     5.48 |                7.47 |           98.67 |             97    |               100 |            95.67 |       0.72 |                   0    |                  4.35 |                 0 | 530.84 |
| noisy       |                100 |                      0    |                     1.45 |                5.72 |           98.33 |             97.67 |               100 |            96    |       0.92 |                   0.01 |                  4.35 |                 0 | 286.45 |
| distractors |                100 |                      0    |                     1.05 |                5.6  |           98.33 |             97.5  |               100 |            95.83 |       0.76 |                   5.71 |                  4.35 |                 0 | 427.64 |
| occlusion   |                100 |                      6.33 |                     1.28 |                5.58 |           98.93 |             97.51 |               100 |            96.44 |       0.93 |                   0    |                  4.35 |                19 | 479.23 |
| hard        |                100 |                      1.33 |                     4.65 |                6.9  |           98.93 |             97.15 |               100 |            96.09 |       0.69 |                   3.99 |                  3.68 |                19 | 492.09 |

## Ablation

### False-positive frames %

| Scenario    |   Full pipeline |   No morphology |   No smoothing |   No min-area filter |
|:------------|----------------:|----------------:|---------------:|---------------------:|
| clear       |            0    |            0    |           0    |                 0    |
| turbid      |            0    |            0    |           0    |                 0    |
| noisy       |            0    |            0    |           0    |                 0    |
| distractors |            0    |            0    |           0    |                 0    |
| occlusion   |            6.33 |            6.33 |           6.33 |                 6.33 |
| hard        |            1.33 |            1.33 |           1.33 |                 6.33 |

### Mean centroid err (px)

| Scenario    |   Full pipeline |   No morphology |   No smoothing |   No min-area filter |
|:------------|----------------:|----------------:|---------------:|---------------------:|
| clear       |            0.96 |            0.94 |           0.96 |                 0.96 |
| turbid      |            5.48 |            5.44 |           5.48 |                 5.48 |
| noisy       |            1.45 |            1.39 |           1.45 |                 1.45 |
| distractors |            1.05 |            1    |           1.05 |                 1.05 |
| occlusion   |            1.28 |            1.23 |           1.28 |                 1.28 |
| hard        |            4.65 |            4.46 |           4.65 |                 4.65 |

### Full cmd acc %

| Scenario    |   Full pipeline |   No morphology |   No smoothing |   No min-area filter |
|:------------|----------------:|----------------:|---------------:|---------------------:|
| clear       |           95.67 |           95.67 |          98.67 |                95.67 |
| turbid      |           95.67 |           95.67 |          97.33 |                95.67 |
| noisy       |           96    |           96    |          98.33 |                96    |
| distractors |           95.83 |           95.83 |          98.67 |                95.83 |
| occlusion   |           96.44 |           96.44 |          99.29 |                96.44 |
| hard        |           96.09 |           96.09 |          97.86 |                95.73 |

### Mask IoU

| Scenario    |   Full pipeline |   No morphology |   No smoothing |   No min-area filter |
|:------------|----------------:|----------------:|---------------:|---------------------:|
| clear       |            0.94 |            0.93 |           0.94 |                 0.94 |
| turbid      |            0.72 |            0.73 |           0.72 |                 0.72 |
| noisy       |            0.92 |            0.87 |           0.92 |                 0.92 |
| distractors |            0.76 |            0.75 |           0.76 |                 0.76 |
| occlusion   |            0.93 |            0.92 |           0.93 |                 0.93 |
| hard        |            0.69 |            0.69 |           0.69 |                 0.69 |

### Spurious blobs/frame

| Scenario    |   Full pipeline |   No morphology |   No smoothing |   No min-area filter |
|:------------|----------------:|----------------:|---------------:|---------------------:|
| clear       |            0    |            0.01 |           0    |                 0    |
| turbid      |            0    |            0.2  |           0    |                 0    |
| noisy       |            0.01 |           25.42 |           0.01 |                 0.01 |
| distractors |            5.71 |            5.77 |           5.71 |                 5.71 |
| occlusion   |            0    |            0.38 |           0    |                 0    |
| hard        |            3.99 |            5.08 |           3.99 |                 3.99 |

### Cmd changes /100 fr

| Scenario    |   Full pipeline |   No morphology |   No smoothing |   No min-area filter |
|:------------|----------------:|----------------:|---------------:|---------------------:|
| clear       |            4.35 |            4.35 |           4.35 |                 4.35 |
| turbid      |            4.35 |            4.35 |           4.35 |                 4.35 |
| noisy       |            4.35 |            4.35 |           4.35 |                 4.35 |
| distractors |            4.35 |            4.35 |           4.35 |                 4.35 |
| occlusion   |            4.35 |            4.35 |           4.35 |                 4.35 |
| hard        |            3.68 |            3.68 |           3.68 |                 5.52 |


## Sensitivity to the saturation threshold (S_min)

| Scenario   |   S_min |   Mask IoU |   Mean centroid err (px) |   Spurious blobs/frame |   False-positive frames % |
|:-----------|--------:|-----------:|-------------------------:|-----------------------:|--------------------------:|
| clear      |      30 |       0.94 |                     1.01 |                   0    |                      0    |
| clear      |      50 |       0.94 |                     1.01 |                   0    |                      0    |
| clear      |      80 |       0.94 |                     0.96 |                   0    |                      0    |
| clear      |     110 |       0.93 |                     0.89 |                   0    |                      0    |
| clear      |     140 |       0.9  |                     0.82 |                   0    |                      0    |
| clear      |     170 |       0.8  |                     1.87 |                   0    |                      0    |
| turbid     |      30 |       0.78 |                     4.46 |                   0    |                      0    |
| turbid     |      50 |       0.78 |                     4.48 |                   0    |                      0    |
| turbid     |      80 |       0.72 |                     5.48 |                   0    |                      0    |
| turbid     |     110 |       0.54 |                     8.92 |                   0    |                      0    |
| turbid     |     140 |       0.01 |                   nan    |                   0.31 |                      0    |
| turbid     |     170 |       0    |                   nan    |                   0    |                      0    |
| hard       |      30 |       0.71 |                     4    |                   3.94 |                      1.33 |
| hard       |      50 |       0.71 |                     4.08 |                   3.94 |                      1.33 |
| hard       |      80 |       0.68 |                     4.63 |                   3.94 |                      1.33 |
| hard       |     110 |       0.61 |                     6.52 |                   2.45 |                      1.33 |
| hard       |     140 |       0.31 |                    11.65 |                   0.02 |                      0.67 |
| hard       |     170 |       0    |                   nan    |                   0    |                      0    |
