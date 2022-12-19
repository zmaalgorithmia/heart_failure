import joblib
import pandas as pd
from pathlib import Path


def load_model(code_dir):
    code_path = Path(code_dir) / Path("random_forest_model.pkl")

    model = joblib.load(code_path)
    return model
