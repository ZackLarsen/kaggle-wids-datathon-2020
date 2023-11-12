""""
save.py

Saves parquet files to disk
"""

import polars as pl
from prefect import flow, get_run_logger


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

    logger.info(f"Saved splits to {cfg.paths.data}")

    return None


@flow
def save_transforms(cfg, X_transformed):
    logger = get_run_logger()
    X_transformed_path = cfg.paths.data.X_transformed
    X_transformed.write_parquet(X_transformed_path)
    logger.info(f"Saved transformed data to {X_transformed_path}")

    return None


@flow
def save_preds(cfg, y_pred):
    logger = get_run_logger()
    y_pred_path = cfg.paths.data.y_pred
    y_pred_pl = pl.DataFrame(y_pred)
    y_pred_pl.write_parquet(y_pred_path)
    logger.info(f"Saved predictions to {y_pred_path}")

    return None


@flow
def save_submission(cfg, formatted_preds):
    logger = get_run_logger()
    submission_path = cfg.paths.data.submission
    formatted_preds.write_csv(submission_path)
    logger.info(f"Saved formatted predictions to {submission_path}")

    return None
