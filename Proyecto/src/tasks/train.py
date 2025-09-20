from prefect import task
from sklearn.metrics import accuracy_score, classification_report

@task
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    import time
    start_time = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    train_time = time.time() - start_time

    return {
        "accuracy": accuracy,
        "precision_1": report['1']['precision'],
        "recall_1": report['1']['recall'],
        "f1_score_1": report['1']['f1-score'],
        "train_time": train_time
    }
    