# src/flow/training_flow.py
from prefect import flow
from src.tasks.train import train_and_evaluate
from src.tasks.evaluate import cross_validate_model

@flow
def training_flow(load_task, split_task, model_builders: dict, label: str = "dataset", load_kwargs: dict | None = None):
    """
    load_task: Prefect task que devuelve (X, y) y acepta **load_kwargs (p.ej., {'name': 'diabetes'})
    split_task: Prefect task (X, y) -> (Xtr, Xte, ytr, yte)
    model_builders: dict nombre -> modelo
    label: para identificar el dataset en los prints
    load_kwargs: kwargs que se pasan a load_task
    """
    load_kwargs = load_kwargs or {}
    print(f"\n=== {label.upper()} ===")
    X, y = load_task(**load_kwargs)            # <-- aquí pasamos los kwargs
    Xtr, Xte, ytr, yte = split_task(X, y)

    for name, model in model_builders.items():
        metrics = train_and_evaluate(model, Xtr, Xte, ytr, yte)
        cv_score = cross_validate_model(model, Xtr, ytr)
        print(f"\n--- {label}:{name} ---")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Precision (1): {metrics['precision_1']:.4f}")
        print(f"Recall (1):    {metrics['recall_1']:.4f}")
        print(f"F1-score (1):  {metrics['f1_score_1']:.4f}")
        print(f"Train time:    {metrics['train_time']:.2f}s")
        print(f"Cross-val:     {cv_score:.4f}")
