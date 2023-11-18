"""
split.py

Splits model_input data into train/validation/test
"""

import polars as pl
from prefect import flow, get_run_logger
from sklearn.model_selection import train_test_split


@flow
def split_train(cfg, data):
    logger = get_run_logger()

    target_var = cfg.target_var
    random_state = cfg.train_test_split.random_state
    train_ratio = cfg.train_test_split.train_ratio
    validation_ratio = cfg.train_test_split.validation_ratio

    X = data.drop(target_var)
    y = data.select(pl.col(target_var))

    logger.info("Splitting data into train and validation sets")
    logger.info(f"train_ratio: {train_ratio}")
    logger.info(f"validation_ratio: {validation_ratio}")
    assert train_ratio + validation_ratio == 1, \
        """train_ratio and validation_ratio must sum to 1"""

    X_train, X_validation, y_train, y_validation = \
        train_test_split(
            X,
            y,
            test_size=validation_ratio,
            random_state=random_state)

    logger.info(f"Size of data: {data.shape}")
    logger.info(f"Size of X_train: {X_train.shape}")
    logger.info(f"Size of X_validation: {X_validation.shape}")

    splits = {
        'X_train': X_train,
        'X_validation': X_validation,
        'y_train': y_train,
        'y_validation': y_validation
    }

    return splits


@flow
def split_test(cfg, data):
    logger = get_run_logger()

    target_var = cfg.target_var

    logger.info("Splitting test set")
    X = data.drop(target_var)
    y = data.select(pl.col(target_var))

    splits = {
        'X_test': X,
        'y_test': y
    }

    return splits


@flow
def split_all(cfg, data):
    logger = get_run_logger()

    target_var = cfg.target_var
    random_state = cfg.train_test_split.random_state
    train_ratio = cfg.train_test_split.train_ratio
    validation_ratio = cfg.train_test_split.validation_ratio
    test_ratio = cfg.train_test_split.test_ratio

    assert train_ratio + validation_ratio + test_ratio == 1, \
        """train_ratio, validation_ratio, and test_ratio must sum to 1"""

    X = data.drop(target_var)
    y = data.select(pl.col(target_var))

    logger.info("Splitting data into train, validation, and test sets")
    logger.info(f"train_ratio: {train_ratio}")
    logger.info(f"validation_ratio: {validation_ratio}")
    logger.info(f"test_ratio: {test_ratio}")

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X,
            y,
            test_size=1 - train_ratio,
            random_state=random_state)

    X_validation, X_test, y_validation, y_test = \
        train_test_split(
            X_test,
            y_test,
            test_size=test_ratio/(test_ratio + validation_ratio),
            random_state=random_state)

    logger.info(f"Size of data: {data.shape}")
    logger.info(f"Size of X_train: {X_train.shape}")
    logger.info(f"Size of X_validation: {X_validation.shape}")
    logger.info(f"Size of X_test: {X_test.shape}")

    splits = {
        'X_train': X_train,
        'X_validation': X_validation,
        'X_test': X_test,
        'y_train': y_train,
        'y_validation': y_validation,
        'y_test': y_test
    }

    return splits
