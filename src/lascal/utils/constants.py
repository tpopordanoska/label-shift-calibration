import os

import numpy as np

# Get the parent directory (move up two levels from the script's location)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
FIGS_DIR = os.path.join(ROOT_DIR, "figs")
LOGITS_DIR = os.path.join(ROOT_DIR, "trained_models", "logits")
LABELS_DIR = os.path.join(ROOT_DIR, "trained_models", "labels")

EPS = 1e-7
STEPS = 100

CAL_METHODS = [
    "uncal",
    "temp_scale_source",
    "vector_scale_source",
    "ensemble_temp_scale",
    "isotonic_regression",
    "cpcs",
    "transcal",
    "head_to_tail",
    "em_alexandari",
    "probability_matching",
    "temp_scale_target",
    "vector_scale_target",
    "lascal"
]
