import polars as pl
from prefect import flow, get_run_logger
from prefect.artifacts import create_table_artifact
from hydra import compose, initialize


@flow
def ingest_raw_data(cfg):
    logger = get_run_logger()
    logger.info("Ingesting raw data")
    raw_path = cfg.paths.data.raw
    raw_data = pl.read_csv(raw_path, infer_schema_length=10000)
    logger.info(f"Raw data shape: {raw_data.shape}")
    raw_data_sample = raw_data.sample(n=5, seed=42)
    raw_data_sample_dict = list(raw_data_sample.to_dicts())
    create_table_artifact(
        key="raw-sample",
        table=raw_data_sample_dict,
        description="Raw Data Sample",
    )

    return raw_data


if __name__ == "__main__":
    with initialize(version_base="1.3.2",
                    config_path="../src/config",
                    job_name="test_flow"):
        cfg = compose(config_name="config")
    raw_data = ingest_raw_data(cfg)
