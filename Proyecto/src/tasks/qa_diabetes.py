# src/tasks/qa_diabetes.py
# -*- coding: utf-8 -*-
from prefect import task
from src.tasks.tunning_diabetes import run_tuning_and_reports_inflow

@task(name="answer-diabetes-questions")
def answer_diabetes_questions(
    csv_path: str = "data/diabetes.csv",
    fairness_threshold: float = 0.05,
    random_state: int = 77,
):
    return run_tuning_and_reports_inflow(
        csv_path=csv_path,
        random_state=random_state,
        fairness_threshold=fairness_threshold,
    )
