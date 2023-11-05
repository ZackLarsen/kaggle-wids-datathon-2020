"""
split_data.py

Splits model_input data into train/validation/test
"""

from pathlib import Path

import polars as pl
from prefect import flow, get_run_logger
from sklearn.model_selection import train_test_split


@flow
def split(cfg, data, random_state=42):
    logger = get_run_logger()
    logger.info("Splitting data into train and test sets")
    target_var = cfg.target_var

    X = data.drop(target_var)
    y = data.select(pl.col(target_var))

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X,
            y,
            test_size=1 - train_ratio,
            random_state=random_state)

    X_val, X_test, y_val, y_test = \
        train_test_split(
            X_test,
            y_test,
            test_size=test_ratio/(test_ratio + validation_ratio),
            random_state=random_state)

    logger.info("Size of data: {data.shape}")
    logger.info("Size of X_train: {X_train.shape}")
    logger.info("Size of X_val: {X_val.shape}")
    logger.info("Size of X_test: {X_test.shape}")

    splits = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

    return splits


@flow
def save_splits(cfg, splits):
    logger = get_run_logger()
    X_train_path = cfg.paths.data.X_train
    X_test_path = cfg.paths.data.X_test
    X_validation_path = cfg.paths.data.X_validation
    y_train_path = cfg.paths.data.y_train
    y_test_path = cfg.paths.data.y_test
    y_validation_path = cfg.paths.data.y_validation

    splits['X_train'].write_parquet(X_train_path)
    splits['X_val'].write_parquet(X_validation_path)
    splits['X_test'].write_parquet(X_test_path)
    splits['y_train'].write_parquet(y_train_path)
    splits['y_val'].write_parquet(y_validation_path)
    splits['y_test'].write_parquet(y_test_path)

    logger.info("Saved splits to {cfg.paths.data}")

    return None
