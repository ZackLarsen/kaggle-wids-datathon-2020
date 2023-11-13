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
from save import save_splits, save_transforms, save_preds, save_submission, save_model


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
    X_validation = splits['X_validation']
    y_validation = splits['y_validation']
    X_test = splits['X_test']
    y_test = splits['y_test']
    X_train_transformed_path = cfg.paths.data.X_train_transformed
    X_validation_transformed_path = cfg.paths.data.X_validation_transformed
    X_test_transformed_path = cfg.paths.data.X_test_transformed
    X_train_transformed = transform(cfg, X=X_train, y=y_train)
    save_transforms(X_train_transformed, X_train_transformed_path)
    X_validation_transformed = transform(cfg, X=X_validation, y=y_validation)
    save_transforms(X_validation_transformed, X_validation_transformed_path)
    X_test_transformed = transform(cfg, X=X_test, y=y_test)
    save_transforms(X_test_transformed, X_test_transformed_path)

    model = train(X_train_transformed, y_train)
    save_model(cfg, model)
    y_pred = predict(model, X_train_transformed)
    save_preds(cfg, y_pred)
    evaluate(y_train, y_pred)
    # plot_threshold(model, X_train_transformed, y_train)
    formatted_preds = format_preds(cfg, X_train_transformed, y_pred)

    # TODO: Run this on the held-out test set
    save_submission(cfg, formatted_preds)
    logger.info("Done!")


if __name__ == "__main__":
    with initialize(version_base="1.3.2",
                    config_path="config",
                    job_name="test_flow"):
        cfg = compose(config_name="config")
        run_flow(cfg)
