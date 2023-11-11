"""
clean_data.py

This script preprocesses the data used for modeling.

Steps:
- Rename columns
- Replace nulls appropriately

"""

from prefect import flow, get_run_logger
from hydra import compose, initialize
import polars as pl
import polars.selectors as cs
import pandas as pd
from skimpy import clean_columns


@flow
def clean(data):
    logger = get_run_logger()
    logger.info("Cleaning data")
    pd_data = data.to_pandas()
    clean_data = clean_columns(pd_data)
    clean_data_pl = pl.from_pandas(clean_data)

    return clean_data_pl
