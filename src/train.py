"""
train.py

trains the model
"""

from prefect import flow, get_run_logger
from sklearn.linear_model import LogisticRegression


@flow
def train(X, y):
    logger = get_run_logger()
    logger.info("Training model")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model
