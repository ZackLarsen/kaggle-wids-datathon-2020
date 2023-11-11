from prefect import flow, task, get_run_logger
from prefect.runtime import flow_run
import polars as pl


def generate_flow_run_name():
    flow_name = flow_run.flow_name
    parameters = flow_run.parameters
    name = parameters["name"]

    return f"{flow_name}-with-{name}"


@task(name="Ingest Data")
def ingest(path):
    logger = get_run_logger()
    logger.info(f"Ingesting data from {path}")
    # data = pl.read_parquet(path)
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    return data


@task(name="Clean Data")
def clean(data):
    clean_data = data
    return clean_data


@task(name="Featurize Data")
def featurize(data):
    featurized_data = data
    return featurized_data


@task(name="Split Data")
def split(data):
    splits = data
    return splits


@task(name="Impute Missing Values")
def impute(data):
    imputed_data = data
    return imputed_data


@task(name="Train Model")
def train(data):
    model = data
    return model


@task(name="Evaluate Model")
def evaluate(data):
    metrics = data
    return metrics


@flow()
def run_flow(path="data/training.parquet"):
    data = ingest(path)
    clean_data = clean(data)
    featurized_data = featurize(clean_data)
    splits = split(featurized_data)
    imputed_data = impute(splits)
    model = train(imputed_data)
    metrics = evaluate(model)
    print(metrics)


if __name__ == "__main__":
    # run_flow.visualize()
    run_flow()
