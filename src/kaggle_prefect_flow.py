# """
# Use Prefect to orchestrate the Kaggle competition pipeline.
# """

# from prefect import flow, task
# import polars as pl
# import pandas as pd
# import mlflow
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import xgboost as xgb


# @task
# def ingest(path):
#     data = pl.read_csv(path)
#     return data


# @task
# def calc_mean_age(data):
#     return data.select("age").mean()


# # @task
# # def transform(data):
# #     # Your code for data transformation
# #     return transformed_data


# @task
# def split(data, test_size=0.2, random_state=42):
#     X = data.drop('hospital_death')
#     y = data['hospital_death']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state)

#     return X_train, X_test, y_train, y_test


# @task
# def save_splits(X_train, X_test, y_train, y_test):
#     X_train.to_parquet('data/X_train.parquet')
#     y_train.to_parquet('data/y_train.parquet')
#     X_test.to_parquet('data/X_test.parquet')
#     y_test.to_parquet('data/y_test.parquet')


# @task
# def train_model(X_train, y_train):
#     model = xgb.XGBClassifier()
#     model.fit(X_train, y_train)
#     return model


# @task
# def predict(model, test_data):
#     preds = model.predict(test_data)
#     return preds


# @task
# def evaluate_model(y_test, y_pred):
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred)

#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 score: {f1:.4f}")
#     print(f"ROC AUC score: {roc_auc:.4f}")

#     return accuracy, precision, recall, f1, roc_auc


# with flow("WiDS2020") as flow:
#     data = ingest('data/training_v2.csv')
#     mean_age = calc_mean_age(data)
#     # transformed_data = transform(data)
#     X_train, X_test, y_train, y_test = split(data)
#     save_splits(X_train, X_test, y_train, y_test)
#     model = train_model(X_train, y_train)
#     preds = predict(model, X_test)

#     accuracy, precision, recall, f1, roc_auc = evaluate_model(y_test, preds)

#     mlflow.set_experiment("first_xgb_clf")
#     with mlflow.start_run():
#         mlflow.log_param("model_type", "XGBoost")
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("precision", precision)
#         mlflow.log_metric("recall", recall)
#         mlflow.log_metric("f1", f1)
#         mlflow.log_metric("roc_auc", roc_auc)

#     flow.run()
