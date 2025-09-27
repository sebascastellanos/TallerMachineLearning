from prefect import task
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# En Pima, estos 0 significan "faltante"
_DIAB_ZERO_AS_NAN = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

@task
def load_diabetes(csv_path: str = "data/diabetes.csv", target: str = "Outcome"):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path.resolve()}. Coloca tu CSV ahí.")
    df = pd.read_csv(path)

    if target not in df.columns:
        raise ValueError(f"El CSV debe contener la columna target '{target}'.")

    # 0 -> NaN en columnas fisiológicamente no-cero (si existen)
    for c in _DIAB_ZERO_AS_NAN:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # fuerza a numérico (si hay strings, se vuelven NaN y luego se imputan)
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X.to_numpy(dtype=float, copy=False), y.to_numpy(dtype=int, copy=False)

@task
def split_and_scale_diabetes(X, y, test_size: float = 0.3, random_state: int = 109):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Imputación necesaria por los 0->NaN
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
