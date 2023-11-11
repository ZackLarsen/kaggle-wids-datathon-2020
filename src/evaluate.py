"""
evaluate.py

evaluates the model
"""

from prefect import flow, get_run_logger
from sklearn.metrics import accuracy_score


@flow
def evaluate(y, y_pred):
    logger = get_run_logger()
    logger.info("Evaluating model")
    accuracy = accuracy_score(y, y_pred)
    logger.info(f"Accuracy: {accuracy}")
    return accuracy
