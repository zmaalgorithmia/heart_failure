import joblib
import pandas as pd
import io
from pathlib import Path


def read_input_data(input_binary_data):
    df = pd.read_csv(io.BytesIO(input_binary_data))
    return df


def load_model(code_dir):
    code_path = Path(code_dir) / Path("random_forest_model.pkl")

    model = joblib.load(code_path)
    return model
