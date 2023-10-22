"""
Use Prefect to orchestrate the Kaggle competition pipeline.
"""

from pathlib import Path
from prefect import flow, get_run_logger
import polars as pl
import polars.selectors as cs
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb


competition_path = Path("/Users/zacklarsen/Documents/Projects/kaggle-wids-datathon-2020/")
mlflow_path = Path(competition_path, "mlruns/")
data_path = Path(competition_path, "data/")
train_path = Path(data_path, "train.parquet")

mlflow.set_tracking_uri(mlflow_path)
mlflow.xgboost.autolog()


@flow
def ingest(path):
    data = pl.read_parquet(path)
    logger = get_run_logger()
    logger.info("Data ingested")

    return data


@flow
def calc_mean_age(data):
    mean_age = data.select(pl.col("age")).mean().item()
    logger = get_run_logger()
    logger.info(f"Mean age calculated as: {mean_age}")

    return mean_age


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
def save_splits(X_train, X_test, y_train, y_test):
    logger = get_run_logger()
    logger.info("Saving splits")
    X_train.write_parquet(Path(data_path, 'X_train.parquet'))
    pl.DataFrame(y_train).write_parquet(Path(data_path, 'y_train.parquet'))
    X_test.write_parquet(Path(data_path, 'X_test.parquet'))
    pl.DataFrame(y_test).write_parquet(Path(data_path, 'y_test.parquet'))

    return None


@flow
def train_model(X_train, y_train):
    logger = get_run_logger()
    logger.info("Training XGBoost classifier")

    with mlflow.start_run() as run:
        model = xgb.XGBClassifier()
        model.fit(X_train.select(cs.numeric()), y_train)

        run_id = run.info.run_id
        # experiment_id = run.info.experiment_id
        model_uri = f"runs:/{run_id}/model"
        model_name = "XGBoost.json"
        mlflow.register_model(model_uri, model_name)

        return model


@flow
def predict(model, test_data):
    logger = get_run_logger()
    logger.info("Predicting on test data")
    preds = model.predict(test_data)

    return preds


@flow
def evaluate_model(y_test, y_pred):
    logger = get_run_logger()
    logger.info("Evaluating model performance")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc


@flow
def prep_workflow(file_path):
    data = ingest(file_path)
    mean_age = calc_mean_age(data)
    X_train, X_test, y_train, y_test = split(data, test_size=0.2, random_state=42)
    save_splits(X_train, X_test, y_train, y_test)
    clf = train_model(X_train.select(cs.numeric()), y_train)
    y_pred = predict(clf, X_test.select(cs.numeric()))
    accuracy, precision, recall, f1, roc_auc = evaluate_model(y_test, y_pred)

    result_dict = {
        'mean_age': mean_age,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    return result_dict
