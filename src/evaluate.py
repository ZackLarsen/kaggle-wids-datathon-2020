"""
evaluate.py

evaluates the model
"""

from prefect import flow, get_run_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from yellowbrick.classifier import DiscriminationThreshold


@flow
def evaluate(y, y_pred):
    logger = get_run_logger()
    logger.info("Evaluating model")
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision score: {precision}")
    logger.info(f"Recall score: {recall}")
    logger.info(f"f1 score: {f1}")
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return metrics


@flow
def plot_threshold(model, X, y):
    visualizer = DiscriminationThreshold(model)
    visualizer.fit(X, y)
    visualizer.show()
