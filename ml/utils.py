# ml/utils.py
import os, pandas as pd
from pathlib import Path
from .synthetic_data import generate_dataset
from .features import batch_extract

def run_full_pipeline(root=".", n=160):
    generate_dataset(root, n=n)
    Path(root+"/data/features").mkdir(parents=True, exist_ok=True)
    batch_extract(Path(root)/"data/keypoints", Path(root)/"data/features/features.csv")
    print("Feature extraction done")
