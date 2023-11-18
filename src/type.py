"""
type.py

Enforces schema for correct data types of
data columns.
"""

import polars as pl
from prefect import flow, get_run_logger


@flow
def enforce_schema(cfg, raw_data: pl.DataFrame) -> pl.DataFrame:
    logger = get_run_logger()
    logger.info("Enforcing schema")
    data_dictionary = pl.read_csv(cfg.paths.data.dictionary)
    data_dictionary = (
        data_dictionary
        .select(pl.col(["Variable Name", "Data Type"]))
        .filter(pl.col("Variable Name") != "icu_admit_type")  # In data dictionary but not in data
        .filter(pl.col("Variable Name") != "pred")  # In data dictionary but not in data
        .with_columns(
            pl.when(pl.col("Variable Name").str.ends_with("_id"))
            .then(pl.lit("string"))
            .otherwise(pl.col("Variable Name"))
        )
    )

    string_cols = (
        data_dictionary
        .filter(pl.col("Data Type") == "string")
        .select(pl.col("Variable Name"))
        .to_series()
        .to_list()
    )

    int_cols = (
        data_dictionary
        .filter(pl.col("Data Type") == "integer")
        .select(pl.col("Variable Name"))
        .to_series()
        .to_list()
    )

    float_cols = (
        data_dictionary
        .filter(pl.col("Data Type") == "numeric")
        .select(pl.col("Variable Name"))
        .to_series()
        .to_list()
    )

    boolean_cols = (
        data_dictionary
        .filter(pl.col("Data Type") == "binary")
        .select(pl.col("Variable Name"))
        .to_series()
        .to_list()
    )

    correctly_typed_data = (
        raw_data
        .with_columns(
            pl.when(pl.col(pl.Utf8) == "NA")
            .then(pl.lit(None))
            .otherwise(pl.col(pl.Utf8))
            .name.keep()
        )
        .with_columns(
            pl.col(string_cols).cast(pl.Utf8),
            pl.col(boolean_cols).cast(pl.Int8),
            pl.col(int_cols).cast(pl.Int32),
            pl.col(float_cols).cast(pl.Float32),
        )
        .with_columns(pl.col(boolean_cols).cast(pl.Boolean))
    )

    return correctly_typed_data
