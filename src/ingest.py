"""
ingest.py

Ingests raw data from disk
"""

import polars as pl
from prefect import flow, get_run_logger
from prefect.artifacts import create_table_artifact


@flow
def ingest_raw_train_data(cfg):
    logger = get_run_logger()
    raw_train_path = cfg.paths.data.raw_train
    logger.info("Ingesting raw train data from {raw_train_path}")
    raw_train = pl.read_csv(raw_train_path, infer_schema_length=10000)
    logger.info(f"Raw train shape: {raw_train.shape}")
    raw_train_sample = raw_train.sample(n=5, seed=42)
    raw_train_sample_dict = list(raw_train_sample.to_dicts())
    create_table_artifact(
        key="raw-train-sample",
        table=raw_train_sample_dict,
        description="Raw Train Sample",
    )
    return raw_train


@flow
def ingest_raw_test_data(cfg):
    logger = get_run_logger()
    raw_test_path = cfg.paths.data.raw_test
    logger.info("Ingesting raw test data from {raw_test_path}")
    raw_test = pl.read_csv(raw_test_path, infer_schema_length=10000)
    logger.info(f"Raw test shape: {raw_test.shape}")
    raw_test_sample = raw_test.sample(n=5, seed=42)
    raw_test_sample_dict = list(raw_test_sample.to_dicts())
    create_table_artifact(
        key="raw-test-sample",
        table=raw_test_sample_dict,
        description="Raw Test Sample",
    )
    return raw_test
