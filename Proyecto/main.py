# main.py
from src.flow.training_flow import training_flow
from src.tasks.cancer import load_dataset, split_and_scale  # tus tasks actuales
from src.tasks.diabetes import load_diabetes, split_and_scale_diabetes  # tus tasks actuales
from src.tasks.iris import load_iris, split_and_scale_iris  # tus tasks actuales
from src.models.knn_model import get_model as get_knn
from src.models.svm_model import get_model as get_svm
from src.models.dt_model  import get_model as get_dt
from src.models.rf_model  import get_model as get_rf

models = {"KNN": get_knn(), "SVM": get_svm(), "DT": get_dt(), "RF": get_rf()}

if __name__ == "__main__":
    # === DIABETES (CSV) ===
    '''training_flow(
        load_task=load_diabetes,
        split_task=split_and_scale_diabetes,
        model_builders=models,
        label="diabetes",
        
    )'''

    # # === CÁNCER (sklearn) ===
    training_flow(
         load_task=load_dataset,
         split_task=split_and_scale,
         model_builders=models,
         label="cancer",
         load_kwargs={"name": "cancer"}
     )
    
    '''training_flow(
        load_task=load_iris,
        split_task=split_and_scale_iris,
        model_builders=models,
        label="iris",
        
    )'''
