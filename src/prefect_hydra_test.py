"""
Use Prefect to orchestrate flows and use Hydra to manage configuration.
"""

# import logging
from pathlib import Path

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
from prefect import flow, get_run_logger

from ingest_data import ingest_raw_data
from clean_data import clean
from split_data import split, save_splits


@flow
@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def run_flow(cfg: DictConfig) -> None:
    logger = get_run_logger()
    random_state = cfg.train_test_split.random_state
    raw_data = ingest_raw_data(cfg)
    # clean_data = clean(raw_data)  # TODO: Add to clean, then split with clean data
    # splits = split(clean_data, random_state)
    splits = split(cfg, raw_data, random_state)
    save_splits(cfg, splits)
    logger.info("Done!")


if __name__ == "__main__":
    with initialize(version_base="1.3.2",
                    config_path="config",
                    job_name="test_flow"):
        cfg = compose(config_name="config")
        # print(OmegaConf.to_yaml(cfg))
        run_flow(cfg)
