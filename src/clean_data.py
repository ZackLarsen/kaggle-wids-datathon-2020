"""
clean_data.py

This script preprocesses the data used for modeling.

Steps:
- Rename columns
- Replace nulls appropriately

"""

from pathlib import Path

from prefect import flow, get_run_logger
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
import polars as pl
import polars.selectors as cs


@flow
def clean(data):
    logger = get_run_logger()
    logger.info("Cleaning data")

    clean_data = data.rename()

    return clean_data
