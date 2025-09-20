# src/flow/training_flow.py
from prefect import flow
from sklearn.base import clone
from src.tasks.train import train_and_evaluate
from src.tasks.evaluate import cross_validate_model, plot_roc  # <-- importa tu función
from math import inf


@flow
def training_flow(load_task, split_task, model_builders: dict, label: str = "dataset", load_kwargs: dict | None = None):
    load_kwargs = load_kwargs or {}
    print(f"\n=== {label.upper()} ===")

    X, y = load_task(**load_kwargs)
    Xtr, Xte, ytr, yte = split_task(X, y)

    fastest_name = None
    fastest_time = inf
    times = {} 

    for name, model in model_builders.items():
        # 1) métricas con tu task
        metrics = train_and_evaluate(model, Xtr, Xte, ytr, yte)
        cv_score = cross_validate_model(model, Xtr, ytr)

        # 2) modelo entrenado para ROC (sin tocar tu task): clonar y ajustar
        fitted_for_plot = clone(model)
        fitted_for_plot.fit(Xtr, ytr)

        # 3) ROC
        plot_roc(fitted_for_plot, Xte, yte, model_name=f"{label}:{name}")

        # 4) prints
        print(f"\n--- {label}:{name} ---")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Precision (1): {metrics['precision_1']:.4f}")
        print(f"Recall (1):    {metrics['recall_1']:.4f}")
        print(f"F1-score (1):  {metrics['f1_score_1']:.4f}")
        print(f"Train time:    {metrics['train_time']:.2f}s")
        print(f"Cross-val:     {cv_score:.4f}")
        
        # --- trackear el más rápido ---
        t = metrics["train_time"]
        times[name] = t
        if t < fastest_time:
            fastest_time = t
            fastest_name = name

    # Al final del loop
    print("\n>>> El modelo más rápido fue:", fastest_name, f"({fastest_time:.2f}s)")
