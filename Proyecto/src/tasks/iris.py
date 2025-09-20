from prefect import task
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer



@task
def load_iris(csv_path: str = "data/iris.csv", target: str | None = None):
    from pathlib import Path
    import numpy as np, pandas as pd

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path.resolve()}. Coloca tu CSV ahí.")
    df = pd.read_csv(path)

    # Resolver target
    orig_target = target
    if target is None:
        target = "Species" if "Species" in df.columns else df.columns[-1]
    elif target not in df.columns:
        # fallback automático si te pasaron uno incorrecto (p. ej., 'Outcome')
        target = "Species" if "Species" in df.columns else df.columns[-1]
        print(f"[load_iris] Aviso: target '{orig_target}' no existe. Usando '{target}'.")

    # Extraer y procesar
    y = df[target]
    X = df.drop(columns=[target]).copy()

    # Asegurar numérico en X
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Codificar y si es texto
    if not np.issubdtype(y.dtype, np.number):
        y = y.astype("category").cat.codes

    return X.to_numpy(dtype=float, copy=False), np.asarray(y, dtype=int)



@task
def split_and_scale_iris(X, y, test_size: float = 0.3, random_state: int = 109):
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
