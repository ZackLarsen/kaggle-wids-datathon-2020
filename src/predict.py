"""
predict.py

Runs the model's predict() method
on unseen data
"""

from prefect import flow, get_run_logger


@flow
def predict(model, X):
    logger = get_run_logger()
    logger.info("Generating predictions")
    y_pred = model.predict(X)
    return y_pred
