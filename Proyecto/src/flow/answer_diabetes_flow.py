'''

# src/flow/answer_diabetes_flow.py
# -*- coding: utf-8 -*-
from prefect import flow
from src.tasks.qa_diabetes import answer_diabetes_questions

@flow(name="answer-diabetes-flow")
def answer_diabetes_flow(
    csv_path: str = "data/diabetes.csv",
    fairness_threshold: float = 0.05,
    random_state: int = 77,
):
    # Ejecuta el task que imprime deltas y classification report
    answer_diabetes_questions.submit(
        csv_path=csv_path,
        fairness_threshold=fairness_threshold,
        random_state=random_state,
    )
'''
