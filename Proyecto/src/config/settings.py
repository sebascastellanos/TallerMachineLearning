from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AVAILABLE_DATASETS = {
    "cancer": "breast_cancer.csv",  # o usa load_breast_cancer
    "diabetes": "diabetes.csv",
    "iris": "iris.csv"
}