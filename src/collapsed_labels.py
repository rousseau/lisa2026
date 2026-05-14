"""Collapsed label mapping for Task 2 segmentation (RUN_0003_COLLAPSED).

Original 12-class labels → collapsed 7-class labels:
  0 → 0 (background)
  1,2 → 1 (Hippocampus L+R)
  3,4 → 2 (Caudate L+R)
  5,6 → 3 (Lentiform L+R)
  7,8 → 4 (Ventricle L+R)
  9,10 → 5 (ExV L+R)
  11 → 6 (Aux)

Reverse mapping for post-processing:
  1 → 1,2 (Hippocampus L,R)
  2 → 3,4 (Caudate L,R)
  3 → 5,6 (Lentiform L,R)
  4 → 7,8 (Ventricle L,R)
  5 → 9,10 (ExV L,R)
  6 → 11 (Aux)
"""

# Forward mapping: original → collapsed
COLLAPSED_MAP = {
    0: 0,
    1: 1, 2: 1,  # Hippocampus
    3: 2, 4: 2,  # Caudate
    5: 3, 6: 3,  # Lentiform
    7: 4, 8: 4,  # Ventricle
    9: 5, 10: 5, # ExV
    11: 6,       # Aux
}

# Reverse mapping: collapsed → list of original labels
REVERSE_MAP = {
    0: [0],
    1: [1, 2],
    2: [3, 4],
    3: [5, 6],
    4: [7, 8],
    5: [9, 10],
    6: [11],
}

# Class names for collapsed labels
COLLAPSED_NAMES = {
    0: "background",
    1: "hippocampus",
    2: "caudate",
    3: "lentiform",
    4: "ventricle",
    5: "exv",
    6: "aux",
}

# Number of collapsed classes
NUM_COLLAPSED_CLASSES = 7
