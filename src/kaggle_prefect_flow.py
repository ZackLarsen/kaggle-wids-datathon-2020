"""
Use Prefect to orchestrate the Kaggle competition pipeline.

Uses Hydra for configuration management.

Uses MLflow for model tracking.
"""

# import logging
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
from prefect import flow, get_run_logger
import polars as pl
import polars.selectors as cs
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb


@flow
def ingest(path):
    data = pl.read_parquet(path)
    logger = get_run_logger()
    logger.info("Data ingested")
    return data


@flow
def clean(data):
    logger = get_run_logger()
    # TODO Add cleaning steps
    logger.info("Data cleaned")

    return data


@flow
def featurize(data):
    logger = get_run_logger()
    # TODO Add featurization steps with featuretools
    logger.info("Data featurized")

    return data


@flow
def split(data, test_size=0.2, random_state=42):
    logger = get_run_logger()
    logger.info("Splitting data into train and test sets")
    X = data.drop('hospital_death')
    y = data['hospital_death']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    logger.info(f"Size of X_train: {X_train.shape}")
    logger.info(f"Size of X_test: {X_test.shape}")
    logger.info(f"Size of y_train: {y_train.shape}")
    logger.info(f"Size of y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


@flow
def save_splits(data_path, X_train, X_test, y_train, y_test):
    logger = get_run_logger()
    logger.info(f"Saving splits to {data_path}")
    X_train.write_parquet(Path(data_path, 'X_train.parquet'))
    pl.DataFrame(y_train).write_parquet(Path(data_path, 'y_train.parquet'))
    X_test.write_parquet(Path(data_path, 'X_test.parquet'))
    pl.DataFrame(y_test).write_parquet(Path(data_path, 'y_test.parquet'))

    return None


@flow
def transform(data):
    logger = get_run_logger()
    logger.info("Transforming data")
    data = clean(data)
    data = featurize(data)

    return data


@flow
def train(mlflow_path, X_train, y_train):
    mlflow.set_tracking_uri(mlflow_path)
    mlflow.xgboost.autolog()
    logger = get_run_logger()
    model = xgb.XGBClassifier()
    model.fit(X_train.select(cs.numeric()), y_train)
    logger.info("XGBoost classifier trained")

    return model


@flow
def register(model):
    # TODO Fix this to use model object or somehow look up its URI
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        # experiment_id = run.info.experiment_id
        model_uri = f"runs:/{run_id}/model"
        model_name = "XGBoost.json"
        mlflow.register_model(model_uri, model_name)


@flow
def predict(model, test_data):
    logger = get_run_logger()
    logger.info("Predicting on test data")
    preds = model.predict(test_data)

    return preds


@flow
def evaluate(y_test, y_pred):
    logger = get_run_logger()
    logger.info("Evaluating model performance")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc


@flow
def run_flow(cfg: DictConfig) -> None:
    logger = get_run_logger()
    data_path = Path(cfg.paths.data)
    train_data_path = Path(cfg.paths.data.train)

    # mlflow.set_tracking_uri(mlflow_path)
    # mlflow.xgboost.autolog()
    # mlflow_path = Path(cfg.paths.mlflow)

    data = ingest(train_data_path)
    logger.info(f"Shape of data: {data.shape}")

    X_train, X_test, y_train, y_test = split(
        data,
        test_size=0.2,
        random_state=42)

    save_splits(data_path, X_train, X_test, y_train, y_test)

    # clf = train(X_train.select(cs.numeric()), y_train)
    # register(clf)
    # y_pred = predict(clf, X_test.select(cs.numeric()))
    # accuracy, precision, recall, f1, roc_auc = evaluate(y_test, y_pred)

    # result_dict = {
    #     'accuracy': accuracy,
    #     'precision': precision,
    #     'recall': recall,
    #     'f1': f1,
    #     'roc_auc': roc_auc
    # }

    # return result_dict
    logger.info("Run completed")


if __name__ == "__main__":
    with initialize(version_base="1.3.2",
                    config_path="config",
                    job_name="test_flow"):
        cfg = compose(config_name="config")
        print(OmegaConf.to_yaml(cfg))
        run_flow(cfg)
