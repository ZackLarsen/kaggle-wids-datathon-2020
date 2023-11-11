"""
Use Prefect to orchestrate flows and use Hydra to manage configuration.
"""

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
from prefect import flow, get_run_logger

from ingest_data import ingest_raw_data
from clean_data import clean
from split_data import split, save_splits
from transform import transform


@flow
@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def run_flow(cfg: DictConfig) -> None:
    logger = get_run_logger()
    raw_data = ingest_raw_data(cfg)
    clean_data = clean(raw_data)
    splits = split(cfg, clean_data)
    save_splits(cfg, splits)
    X_train = splits['X_train']
    y_train = splits['y_train']
    transform(cfg, X=X_train, y=y_train)
    logger.info("Done!")


if __name__ == "__main__":
    with initialize(version_base="1.3.2",
                    config_path="config",
                    job_name="test_flow"):
        cfg = compose(config_name="config")
        # print(OmegaConf.to_yaml(cfg))
        run_flow(cfg)
