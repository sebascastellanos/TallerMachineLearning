from sklearn.model_selection import cross_val_score
from prefect import task

@task
def cross_validate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return scores.mean()
