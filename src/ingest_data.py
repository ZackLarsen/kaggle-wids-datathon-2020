import polars as pl
from hydra import compose, initialize


def ingest_raw_data(cfg):
    raw_path = cfg.paths.data.raw
    raw_data = pl.read_csv(raw_path, infer_schema_length=10000)

    return raw_data


if __name__ == "__main__":
    with initialize(version_base="1.3.2",
                    config_path="../src/config",
                    job_name="test_flow"):
        cfg = compose(config_name="config")
    raw_data = ingest_raw_data(cfg)
