"""
Use Prefect to orchestrate flows and use Hydra to manage configuration.
"""

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from prefect import flow, get_run_logger

from ingest import ingest_raw_data
from type import enforce_schema
from clean import clean
from split import split
from transform import transform
from train import train
from evaluate import evaluate, plot_threshold
from predict import predict
from format import format_preds
from save import save_splits, save_transforms, save_preds, save_submission


@flow
@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def run_flow(cfg: DictConfig) -> None:
    logger = get_run_logger()
    raw_data = ingest_raw_data(cfg)
    enforce_schema(cfg, raw_data)
    clean_data = clean(raw_data)
    splits = split(cfg, clean_data)
    save_splits(cfg, splits)
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_transformed = transform(cfg, X=X_train, y=y_train)
    save_transforms(cfg, X_transformed)
    model = train(X_transformed, y_train)
    y_pred = predict(model, X_transformed)
    save_preds(cfg, y_pred)
    evaluate(y_train, y_pred)
    plot_threshold(model, X_transformed, y_train)
    formatted_preds = format_preds(cfg, X_transformed, y_pred)
    save_submission(cfg, formatted_preds)
    logger.info("Done!")


if __name__ == "__main__":
    with initialize(version_base="1.3.2",
                    config_path="config",
                    job_name="test_flow"):
        cfg = compose(config_name="config")
        run_flow(cfg)
