"""
split_data.py

Splits model_input data into train/validation/test
"""

from pathlib import Path

from prefect import flow, get_run_logger
from sklearn.model_selection import train_test_split

train_path = Path("/Users/zacklarsen/Documents/Projects/kaggle-wids-datathon-2020/data/train.parquet")


@flow
def split(data, test_size=0.2, random_state=42):
    logger = get_run_logger()
    logger.info("Splitting data into train and test sets")
    X = data.drop('hospital_death')
    y = data['hospital_death']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    logger.info("Size of train set: {X_train.shape}")
    logger.info("Size of test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test
