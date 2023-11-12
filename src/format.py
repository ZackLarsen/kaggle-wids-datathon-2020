"""
format.py

Formats predictions for submission to Kaggle.
"""

import polars as pl
from prefect import flow, get_run_logger


@flow
def format_preds(cfg, X_transformed, y_pred):
    logger = get_run_logger()
    logger.info("Formatting predictions for submission")
    X_transformed_path = cfg.paths.data.X_transformed
    y_pred_path = cfg.paths.data.y_pred
    X_transformed = pl.read_parquet(X_transformed_path)
    y_pred = pl.read_parquet(y_pred_path)
    assert len(X_transformed) == len(y_pred), "Lengths of X_transformed and y_pred do not match"
    X_transformed = X_transformed.with_row_count()
    y_pred = y_pred.with_row_count()
    formatted_preds = X_transformed.join(y_pred, left_on='row_nr', right_on='row_nr', how='inner')
    formatted_preds = (
        formatted_preds
        .select(['row_nr', 'column_0'])
        .rename(
            {'row_nr': 'encounter_id',
             'column_0': 'hospital_death'
             }
        )
    )

    return formatted_preds
