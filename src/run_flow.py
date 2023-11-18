"""
Prefect is used to orchestrate flows.

Hydra is used to manage configuration.

Individual Python scripts perform specific functions
in the style of MLflow's Recipes.
"""

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from prefect import flow, get_run_logger

from ingest import ingest_raw_train_data, ingest_raw_test_data
from type import enforce_schema
from clean import clean
from split import split
from transform import transform
from train import train
from evaluate import evaluate, plot_threshold
from predict import predict
from format import format_preds
from save import save_typed_data, save_splits, save_transforms, save_preds, save_submission, save_model


@flow
@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def run_flow(cfg: DictConfig) -> None:
    logger = get_run_logger()
    # TODO: Validate config
    logger.info("Starting flow")
    train_raw = ingest_raw_train_data(cfg)
    train_typed = enforce_schema(cfg, train_raw)
    train_typed_path = cfg.paths.data.train_typed
    save_typed_data(cfg, train_typed, train_typed_path)
    test_raw = ingest_raw_test_data(cfg)
    test_typed = enforce_schema(cfg, test_raw)
    test_typed_path = cfg.paths.data.test_typed
    save_typed_data(cfg, test_typed, test_typed_path)
    train_data_cleaned = clean(train_typed)
    test_data_cleaned = clean(test_typed)
    # TODO: Save cleaned data
    splits = split(cfg, train_data_cleaned)
    save_splits(cfg, splits)
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_validation = splits['X_validation']
    y_validation = splits['y_validation']
    # TODO: Move test split out of splits since we have unlabeled.csv
    X_test = splits['X_test']
    y_test = splits['y_test']
    X_train_transformed_path = cfg.paths.data.X_train_transformed
    X_validation_transformed_path = cfg.paths.data.X_validation_transformed
    X_test_transformed_path = cfg.paths.data.X_test_transformed
    # TODO: Separate creation of transformation pipeline from fit()
    # TODO: Check if a pipeline defined using train data should not
    # be used on test data - any leakage? See https://stackoverflow.com/questions/68284264/does-the-pipeline-object-in-sklearn-transform-the-test-data-when-using-the-pred

    X_train_transformed = transform(cfg, X=X_train, y=y_train)
    save_transforms(X_train_transformed, X_train_transformed_path)
    X_validation_transformed = transform(cfg, X=X_validation, y=y_validation)
    save_transforms(X_validation_transformed, X_validation_transformed_path)
    X_test_transformed = transform(cfg, X=X_test, y=y_test)
    save_transforms(X_test_transformed, X_test_transformed_path)

    model = train(X_train_transformed, y_train)
    save_model(cfg, model)
    # TODO: Register model with MLflow tracking server
    # register(cfg, model)
    y_pred = predict(model, X_train_transformed)
    save_preds(cfg, y_pred)
    evaluate(y_train, y_pred)
    # plot_threshold(model, X_train_transformed, y_train)
    # TODO: Plot explainer dashboard

    # TODO: Run cross-validation

    # TODO: Run this on the held-out test set
    formatted_preds = format_preds(cfg, X_test_transformed, y_pred)
    save_submission(cfg, formatted_preds)
    logger.info("Done!")


if __name__ == "__main__":
    with initialize(version_base="1.3.2",
                    config_path="config",
                    job_name="test_flow"):
        cfg = compose(config_name="config")
        run_flow(cfg)
