# -*- coding: utf-8 -*-
"""
Script: tuning_diabetes.py
Responde, con código ejecutable, a:
  1) ¿Cuánto afectó el tuning de hiperparámetros el desempeño de SVM y kNN?
  2) Según el classification report del mejor modelo, ¿todas las clases
     se predicen igual de bien?

Imprime:
- Líneas "Modelo: <nombre> | Accuracy: <acc> | F1-macro: <f1>"
- Deltas antes vs. después de tuning
- Classification report y matriz de confusión del mejor modelo
- Juicio automático sobre equidad entre clases (por F1 por-clase)
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# --------------------------
# Utilidades
# --------------------------
def _load_xy_from_csv(path: str, target_col: Optional[str] = "Outcome"):
    df = pd.read_csv(path)
    if target_col is not None and target_col in df.columns:
        y = df[target_col].to_numpy()
        X = df.drop(columns=[target_col]).to_numpy()
    else:
        y = df.iloc[:, -1].to_numpy()
        X = df.iloc[:, :-1].to_numpy()
    return X, y


@dataclass
class EvalResult:
    name: str
    acc: float
    f1m: float
    y_pred: np.ndarray


def _eval_model(name: str, estimator, X_train, y_train, X_test, y_test) -> EvalResult:
    t0 = perf_counter()
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    dt = perf_counter() - t0
    print(f"Modelo: {name} | Accuracy: {acc:.3f} | F1-macro: {f1m:.3f} (fit+pred {dt:.2f}s)")
    return EvalResult(name=name, acc=acc, f1m=f1m, y_pred=y_pred)


def _delta_str(after: EvalResult, before: EvalResult) -> str:
    return f"ΔAcc={after.acc - before.acc:+.3f} | ΔF1={after.f1m - before.f1m:+.3f}"


# --------------------------
# Núcleo: tuning y comparación
# --------------------------
def run_tuning_and_reports(
    data_path: str = "data/diabetes.csv",
    target_col: Optional[str] = "Outcome",
    test_size: float = 0.25,
    random_state: int = 77,
    cv_folds: int = 5,
    fairness_threshold: float = 0.05,  # tolerancia de diferencia de F1 por clase
):
    # 1) Datos
    X, y = _load_xy_from_csv(data_path, target_col)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) Modelos baseline (con escalado adecuado)
    knn_base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ])
    svm_base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

    print("\n== Baselines (antes de tuning) ==")
    knn_before = _eval_model("kNN(k=7) BASE", knn_base, Xtr, ytr, Xte, yte)
    svm_before = _eval_model("SVM-RBF BASE", svm_base, Xtr, ytr, Xte, yte)

    # 3) Tuning con GridSearchCV
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
    knn_grid = {
        "clf__n_neighbors": [3,5,7,9,11,15],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]  # Manhattan / Euclidiana
    }
    knn_gs = GridSearchCV(
        knn_pipe, knn_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
    )

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf"))
    ])
    svm_grid = {
        "clf__C": [0.1, 1, 3, 10],
        "clf__gamma": ["scale", 0.01, 0.1, 0.3, 1.0]
    }
    svm_gs = GridSearchCV(
        svm_pipe, svm_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
    )

    print("\n== Tuning (GridSearchCV) ==")
    knn_after = _eval_model("kNN TUNED", knn_gs, Xtr, ytr, Xte, yte)
    print(f"  Mejor KNN params: {knn_gs.best_params_}")
    svm_after = _eval_model("SVM-RBF TUNED", svm_gs, Xtr, ytr, Xte, yte)
    print(f"  Mejor SVM params: {svm_gs.best_params_}")

    # 4) Respuesta 1: ¿Cuánto afectó el tuning?
    print("\n== Efecto del tuning (después - antes) ==")
    print(f"- kNN: {_delta_str(knn_after, knn_before)}")
    print(f"- SVM: {_delta_str(svm_after, svm_before)}")

    # 5) Elegir mejor modelo (por F1-macro; si empata, por accuracy)
    candidates = [("kNN TUNED", knn_gs.best_estimator_, knn_after),
                  ("SVM-RBF TUNED", svm_gs.best_estimator_, svm_after)]
    candidates.sort(key=lambda t: (t[2].f1m, t[2].acc), reverse=True)
    best_name, best_est, best_eval = candidates[0]

    # 6) Respuesta 2: classification report y equidad por clase
    print(f"\n== Mejor modelo: {best_name} ==")
    y_pred = best_eval.y_pred
    print("Classification report:")
    print(classification_report(yte, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(yte, y_pred))

    # Heurística: “¿todas las clases se predicen igual de bien?”
    # Usamos F1 por clase y comparamos su dispersión vs. un umbral.
    report_dict: Dict[str, Any] = _report_to_dict(yte, y_pred)
    f1_per_class = np.array([v["f1-score"] for k, v in report_dict.items() if k.isdigit()])
    f1_range = f1_per_class.max() - f1_per_class.min()
    equal_msg = "SÍ" if f1_range <= fairness_threshold else "NO"
    print(f"\n¿Todas las clases se predicen igual de bien? {equal_msg} "
          f"(dif. máx F1 entre clases = {f1_range:.3f}, umbral = {fairness_threshold:.3f})")

    return {
        "before": {"knn": knn_before, "svm": svm_before},
        "after": {"knn": knn_after, "svm": svm_after},
        "best_model_name": best_name,
        "best_estimator": best_est,
        "f1_gap_between_classes": float(f1_range),
        "fair_across_classes": bool(f1_range <= fairness_threshold),
    }


def _report_to_dict(y_true, y_pred) -> Dict[str, Dict[str, float]]:
    """Devuelve el classification_report como dict sin imprimir."""
    from sklearn.metrics import classification_report as _cr
    return _cr(y_true, y_pred, output_dict=True, digits=3)


# --------------------------
# CLI
# --------------------------
def ejecutar():
    import argparse
    parser = argparse.ArgumentParser(description="Tuning kNN y SVM en diabetes.csv con reporte por clase")
    parser.add_argument("--data_path", type=str, default="data/diabetes.csv")
    parser.add_argument("--target_col", type=str, default="Outcome")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=77)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--fairness_threshold", type=float, default=0.05)
    args = parser.parse_args()

    run_tuning_and_reports(
        data_path=args.data_path,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        fairness_threshold=args.fairness_threshold
    )


if __name__ == "__main__":
    ejecutar()
