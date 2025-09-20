from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from prefect import task

@task
def load_dataset(name: str):
    if name == "cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        raise ValueError("Dataset no soportado aún.")
    return X, y

@task
def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
