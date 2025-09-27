# -*- coding: utf-8 -*-
"""
Archivo: compare_scaling_diabetes.py
Propósito: comparar kNN(k=7) y SVM(RBF) CON vs SIN escalado usando data/diabetes.csv

Expone:
    run_scaling_comparison_diabetes(...): función reutilizable
    ejecutar(): "main" del script (no se llama main) para correr desde terminal

Formato de impresión:
'Modelo: <nombre> | Accuracy: <acc> | F1-macro: <f1>'
"""

from time import perf_counter
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def _load_xy_from_csv(path: str, target_col: Optional[str] = "Outcome"):
    """
    Carga un CSV y separa X, y.
    - Si target_col existe, la usa.
    - Si no, asume que la última columna es la etiqueta.
    """
    df = pd.read_csv(path)
    if target_col is not None and target_col in df.columns:
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
    return X, y


def _eval_model(name, estimator, X_train, y_train, X_test, y_test):
    t0 = perf_counter()
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    # Impresión en el formato preferido
    print(f"Modelo: {name} | Accuracy: {acc:.3f} | F1-macro: {f1m:.3f}")
    return {"name": name, "acc": acc, "f1": f1m}


def run_scaling_comparison_diabetes(
    data_path: str = "dada/diabetes.csv",
    target_col: Optional[str] = "Outcome",
    test_size: float = 0.25,
    random_state: int = 77
):
    """
    Ejecuta la comparación en diabetes.csv y devuelve los resultados en una lista de dicts.
    Imprime métricas en el formato solicitado.
    """
    # 1) Datos
    X, y = _load_xy_from_csv(data_path, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = []

    # 2) kNN (k=7) sin y con escalado
    knn_raw = KNeighborsClassifier(n_neighbors=7)
    knn_scaled = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ])

    # 3) SVM (RBF) sin y con escalado
    svm_raw = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm_scaled = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ])

    # 4) Evaluaciones (cada una imprime una línea)
    results.append(_eval_model("kNN sin escalado", knn_raw, X_train, y_train, X_test, y_test))
    results.append(_eval_model("kNN con escalado", knn_scaled, X_train, y_train, X_test, y_test))
    results.append(_eval_model("SVM-RBF sin escalado", svm_raw, X_train, y_train, X_test, y_test))
    results.append(_eval_model("SVM-RBF con escalado", svm_scaled, X_train, y_train, X_test, y_test))

    # 5) Conclusión corta
    def pair(prefix_a: str, prefix_b: str):
        a = next(r for r in results if r["name"] == prefix_a)
        b = next(r for r in results if r["name"] == prefix_b)
        return a, b

    knn_sin, knn_con = pair("kNN sin escalado", "kNN con escalado")
    svm_sin, svm_con = pair("SVM-RBF sin escalado", "SVM-RBF con escalado")

    delta_knn_acc = knn_con["acc"] - knn_sin["acc"]
    delta_knn_f1  = knn_con["f1"] - knn_sin["f1"]
    delta_svm_acc = svm_con["acc"] - svm_sin["acc"]
    delta_svm_f1  = svm_con["f1"] - svm_sin["f1"]

    print("\nConclusión:")
    print(f"- Para kNN(k=7), escalar cambió Accuracy {delta_knn_acc:+.3f} y F1-macro {delta_knn_f1:+.3f}.")
    print(f"- Para SVM(RBF), escalar cambió Accuracy {delta_svm_acc:+.3f} y F1-macro {delta_svm_f1:+.3f}.")
    print("Motivo: ambos son sensibles a las escalas (distancias/productos internos). Estándarizar con StandardScaler suele mejorar estabilidad y desempeño.\n")

    return results


# =========================
# "Main" del script (NO se llama main)
# =========================
def ejecutar(): 
    """
    Punto de entrada del script (en lugar de 'main').
    Permite pasar argumentos por CLI, pero llama internamente a run_scaling_comparison_diabetes.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Comparar escalado en kNN(k=7) y SVM(RBF) con diabetes.csv")
    parser.add_argument("--data_path", type=str, default="data/diabetes.csv")
    parser.add_argument("--target_col", type=str, default="Outcome")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=77)
    args = parser.parse_args()

    run_scaling_comparison_diabetes(
        data_path=args.data_path,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state
    )


# Ejecuta solo si corres este archivo directamente
if __name__ == "__main__":
    ejecutar()
