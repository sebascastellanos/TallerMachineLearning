from sklearn.svm import SVC

def get_model():
    # Kernel radial básico, se puede tunear con C y gamma
    return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
