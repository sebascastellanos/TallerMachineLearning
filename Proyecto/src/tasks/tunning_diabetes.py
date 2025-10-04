# src/tasks/tuning_diabetes.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def compute_tuning_effects(
    csv_path: str = "data/diabetes.csv",
    target_col: str = "Outcome",
    random_state: int = 77,
    fairness_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Calcula, SIN imprimir, el efecto del tuning en kNN y SVM y analiza si las
    clases se predicen igual de bien para el MEJOR modelo (por F1-macro).

    Devuelve:
      {
        'deltas': {'knn': {'acc': Δ, 'f1': Δ, 'best_params': {...}},
                   'svm': {'acc': Δ, 'f1': Δ, 'best_params': {...}} },
        'best_model_name': 'kNN TUNED' | 'SVM-RBF TUNED',
        'classification_report_str': str,
        'confusion_matrix': np.ndarray,
        'fair_across_classes': bool,
        'f1_gap_between_classes': float
      }
    """
    df = pd.read_csv(csv_path)
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col]).to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    # Baselines con escalado
    knn_base = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))])
    svm_base = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))])

    def _eval(est):
        est.fit(Xtr, ytr)
        yp = est.predict(Xte)
        return accuracy_score(yte, yp), f1_score(yte, yp, average="macro"), yp

    acc_kb, f1_kb, _       = _eval(knn_base)
    acc_sb, f1_sb, _       = _eval(svm_base)

    # Tuning con GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    knn_gs = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors":[3,5,7,9,11,15], "clf__weights":["uniform","distance"], "clf__p":[1,2]},
        scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
    )
    svm_gs = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
        {"clf__C":[0.1,1,3,10], "clf__gamma":["scale",0.01,0.1,0.3,1.0]},
        scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
    )

    acc_ka, f1_ka, yp_knn  = _eval(knn_gs)
    acc_sa, f1_sa, yp_svm  = _eval(svm_gs)

    deltas = {
        "knn": {"acc": float(acc_ka - acc_kb), "f1": float(f1_ka - f1_kb), "best_params": knn_gs.best_params_},
        "svm": {"acc": float(acc_sa - acc_sb), "f1": float(f1_sa - f1_sb), "best_params": svm_gs.best_params_},
    }

    # Mejor por F1, luego Accuracy
    best_name, best_pred = ("kNN TUNED", yp_knn) if (f1_ka, acc_ka) >= (f1_sa, acc_sa) else ("SVM-RBF TUNED", yp_svm)

    # Classification report y equidad por clase (gap de F1 entre clases)
    report_str  = classification_report(yte, best_pred, digits=3)
    report_dict = classification_report(yte, best_pred, output_dict=True, digits=3)

    f1_class = []
    for k, v in report_dict.items():
        if isinstance(v, dict) and (k.isdigit() or k.replace('.', '', 1).isdigit()):
            if "f1-score" in v:
                f1_class.append(v["f1-score"])
    f1_class = np.array(f1_class, dtype=float) if len(f1_class) else np.array([0.0])
    f1_gap   = float(np.nanmax(f1_class) - np.nanmin(f1_class))
    fair     = bool(f1_gap <= fairness_threshold)

    cm = confusion_matrix(yte, best_pred)

    return {
        "deltas": deltas,
        "best_model_name": best_name,
        "classification_report_str": report_str,
        "confusion_matrix": cm,
        "fair_across_classes": fair,
        "f1_gap_between_classes": f1_gap,
    }


def run_tuning_and_reports_inflow(
    csv_path: str = "data/diabetes.csv",
    target_col: str = "Outcome",
    random_state: int = 77,
    fairness_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Wrapper para usar dentro de un task/flow: calcula e IMPRIME las respuestas.
    """
    res = compute_tuning_effects(
        csv_path=csv_path,
        target_col=target_col,
        random_state=random_state,
        fairness_threshold=fairness_threshold,
    )

    d = res["deltas"]
    print("\n== (Q1) Efecto del tuning ==")
    print(f"- kNN: ΔAcc={d['knn']['acc']:+.3f} | ΔF1={d['knn']['f1']:+.3f} | best={d['knn']['best_params']}")
    print(f"- SVM: ΔAcc={d['svm']['acc']:+.3f} | ΔF1={d['svm']['f1']:+.3f} | best={d['svm']['best_params']}")

    print(f"\n== (Q2) Mejor modelo: {res['best_model_name']} ==")
    print("Classification report:")
    print(res["classification_report_str"])
    print(
        "¿Todas las clases se predicen igual de bien?:",
        "SÍ" if res["fair_across_classes"] else "NO",
        f"(gap F1={res['f1_gap_between_classes']:.3f})",
    )
    return res
