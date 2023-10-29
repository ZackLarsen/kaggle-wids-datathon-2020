"""
Use Prefect to orchestrate flows and use Hydra to manage configuration.
"""

# import logging
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
from prefect import flow, get_run_logger, Parameter
import polars as pl
import polars.selectors as cs
from sklearn.model_selection import train_test_split


@flow
def ingest(path):
    data = pl.read_parquet(path)
    logger = get_run_logger()
    logger.info("Data ingested")
    return data


@flow
def split(data, test_size=0.2, random_state=42):
    logger = get_run_logger()
    logger.info("Splitting data into train and test sets")
    X = data.drop('hospital_death')
    y = data['hospital_death']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    logger.info("Size of train set: {X_train.shape}")
    logger.info("Size of test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


@flow
def save_splits(data_path, X_train, X_test, y_train, y_test):
    logger = get_run_logger()
    logger.info("Saving splits")
    X_train.write_parquet(Path(data_path, 'X_train.parquet'))
    pl.DataFrame(y_train).write_parquet(Path(data_path, 'y_train.parquet'))
    X_test.write_parquet(Path(data_path, 'X_test.parquet'))
    pl.DataFrame(y_test).write_parquet(Path(data_path, 'y_test.parquet'))

    return None


@flow
@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def run_flow(cfg: DictConfig) -> None:
    logger = get_run_logger()
    train_data_path = Path(cfg.paths.data.train)
    train = pl.read_parquet(train_data_path)
    print(train.shape)
    logger.info("Info level message")

    # mlflow.set_tracking_uri(mlflow_path)
    # mlflow.xgboost.autolog()
    # train_data_path = Path(cfg.paths.data.train)
    # mlflow_path = Path(cfg.paths.mlflow)
    # data = ingest(train_data_path)
    # X_train, X_test, y_train, y_test = split(
    #     data,
    #     test_size=0.2,
    #     random_state=42)
    # save_splits(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run_flow()
    test_data_ratio = Parameter("test_data_ratio", default=0.2)