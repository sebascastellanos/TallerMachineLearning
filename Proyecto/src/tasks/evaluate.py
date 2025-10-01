from sklearn.model_selection import cross_val_score
from prefect import task

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt

# @task
def plot_roc(model, X_test, y_test, model_name="Modelo"):
    """
    Grafica ROC para binario o multiclase (OvR), soportando:
    - predict_proba -> (n_samples, n_classes)
    - decision_function -> (n_samples,) en binario o (n_samples, n_classes) en multiclase
    """
    # 1) Obtener puntajes
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)  # (n, C)
    else:
        scores = model.decision_function(X_test)  # (n,) o (n, C)
        # Estandarizar a (n, C)
        if scores.ndim == 1:  # binario
            # columna 1 = clase positiva, columna 0 = negativa
            y_score = np.column_stack([-scores, scores])
        else:
            y_score = scores  # ya es (n, C)

    classes = np.unique(y_test)
    n_classes = len(classes)

    plt.figure()

    if n_classes == 2:
        # 2) Binario: usar columna de la clase positiva
        # Por convención, tratamos a la clase "1" como positiva si existe; si no, la mayor.
        pos_label = 1 if 1 in classes else classes.max()
        # Si el modelo tiene classes_, intenta alinear con su orden
        pos_index = None
        if hasattr(model, "classes_"):
            classes_model = list(model.classes_)
            if pos_label in classes_model:
                pos_index = classes_model.index(pos_label)

        if pos_index is None:
            # fallback: asumir columna 1 corresponde a positiva
            pos_index = 1 if y_score.ndim == 2 and y_score.shape[1] >= 2 else 0

        fpr, tpr, _ = roc_curve((y_test == pos_label).astype(int), y_score[:, pos_index])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

    else:
        # 3) Multiclase OvR
        y_test_bin = label_binarize(y_test, classes=classes)  # (n, C)
        C = y_test_bin.shape[1]
        for i in range(C):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} clase {classes[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'Curva ROC - {model_name}')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

@task
def cross_validate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return scores.mean()
