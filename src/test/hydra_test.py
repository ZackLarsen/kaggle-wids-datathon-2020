# import os
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import polars as pl


log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def my_app(cfg: DictConfig) -> None:
    # working_dir = os.getcwd()
    # print(f"The current working directory is {working_dir}")
    train_data_path = Path(cfg.paths.data.train)
    train = pl.read_parquet(train_data_path)
    print(train.shape)
    log.info("Info level message")


if __name__ == "__main__":
    my_app()
