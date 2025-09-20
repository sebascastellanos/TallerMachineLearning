from prefect import flow
from src.tasks.preprocessing import load_dataset, split_and_scale
from src.models.knn_model import get_model as get_knn
from src.models.svm_model import get_model as get_svm
from src.models.dt_model import get_model as get_dt
from src.models.rf_model import get_model as get_rf
from src.tasks.train import train_and_evaluate
from src.tasks.evaluate import cross_validate_model

@flow
def training_flow(dataset_name: str = "cancer"):
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    models = {
        "KNN": get_knn(),
        "SVM": get_svm(),
        "DT": get_dt(),
        "RF": get_rf()
    }

    for name, model in models.items():
        metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        cv_score = cross_validate_model(model, X_train, y_train)

        print(f"\n--- {name} ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (1): {metrics['precision_1']:.4f}")
        print(f"Recall (1): {metrics['recall_1']:.4f}")
        print(f"F1-score (1): {metrics['f1_score_1']:.4f}")
        print(f"Train time: {metrics['train_time']:.2f}s")
        print(f"Cross-validation: {cv_score:.4f}")
