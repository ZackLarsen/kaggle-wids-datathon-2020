"""
transform.py

builds a pipeline for preprocessing data
"""

import polars as pl
from prefect import flow, get_run_logger
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


@flow
def transform(cfg, X, y):
    logger = get_run_logger()
    logger.info("Transforming input data")

    logger.info("Defining numerical and categorical pipelines")
    numeric_features = list(cfg.numeric_features)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = list(cfg.categorical_features)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    logger.info("Combining transformers into a ColumnTransformer")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    return pipeline


@flow
def fit_transform(pipeline, X, y):
    logger = get_run_logger()
    logger.info("Fitting the pipeline on the input data")
    pipeline.fit(X.to_pandas(), y.to_pandas())
    logger.info("Applying transformations using fitted pipeline")
    X_preprocessed = pipeline.transform(X.to_pandas())
    X_preprocessed = pl.DataFrame(X_preprocessed.toarray())
    X_preprocessed.columns = list(pipeline.get_feature_names_out())

    return X_preprocessed
