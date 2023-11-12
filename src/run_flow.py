"""
Use Prefect to orchestrate flows and use Hydra to manage configuration.
"""

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from prefect import flow, get_run_logger

from ingest import ingest_raw_data
from clean import clean
from split import split, save_splits
from transform import transform
from train import train
from evaluate import evaluate
from predict import predict


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
    X_transformed = transform(cfg, X=X_train, y=y_train)
    model = train(X_transformed, y_train)
    y_pred = predict(model, X_transformed)
    evaluate(y_train, y_pred)
    logger.info("Done!")


if __name__ == "__main__":
    with initialize(version_base="1.3.2",
                    config_path="config",
                    job_name="test_flow"):
        cfg = compose(config_name="config")
        run_flow(cfg)
