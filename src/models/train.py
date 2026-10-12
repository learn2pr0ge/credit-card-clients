import json
import os
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from pipeline import create_pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

path = Path(__file__).resolve().parent
path = str(path)
path = path[:-10]
print(path)


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(f"{path}data/processed/credit-scoring-model-ready_for_model.csv")
        return df
    except FileNotFoundError:
        print("Файл датасета не найден")
        return sys.exit(1)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Функция запуска random_forrest с подбором гиперпараметров через GridSearch
result_dict = {}


def mlfow_run_rand_forrest(X_train, X_test, y_train, y_test, model="random_forest"):
    numeric_features = [
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "Avg_exp_5",
        "Avg_exp_4",
        "Avg_exp_3",
        "Avg_exp_2",
        "Avg_exp_1",
    ]
    categorical_features = [
        "PAY_1",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "SE_MA",
        "AgeBin",
        "SE_AG",
        "Client_6",
        "Client_5",
        "Client_4",
        "Client_3",
        "Client_2",
        "Client_1",
    ]

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("credit-scoring-model")
    pipeline = create_pipeline(numeric_features, categorical_features)
    # Добавляем GridSearch для подбора гиперпараметров
    # Тут сразу лучшие параметры чтобы вам не ждать всего перебора
    param_grid = {
        "classifier__n_estimators": [400],
        "classifier__criterion": ["entropy"],
        "classifier__max_depth": [15],
        "classifier__min_samples_leaf": [10],
    }

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    with mlflow.start_run(run_name="credit-scoring-model"):

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_params(pipeline["classifier"].get_params())
        print("Запуск pipeline Random Forrest с GridSearchCV")
        grid_search.fit(X_train, y_train)
        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучший ROC-AUC на кросс-валидации: {grid_search.best_score_}")

        best_model = grid_search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        # Подсчет метрик

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Метрика accuracy Random Forrest c GridSearchCV: {accuracy:.4f}")
        print(f"Метрика precision Random Forrest c GridSearchCV: {precision:.4f}")
        print(f"Метрика recall Random Forrest c GridSearchCV: {recall:.4f}")
        print(f"Метрика f1-score Random Forrest c GridSearchCV: {f1:.4f}")
        print(f"Метрика ROC-AUC Random Forrest c GridSearchCV: {auc:.4f}")
        result_dict["Random_Forrest"] = round(auc, 4)
        # Логирование метрик
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_roc_auc", grid_search.best_score_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)

        # Построение графика ROC-кривой

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        plt.figure(figsize=(16, 10))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC кривая (area = %0.2f)" % auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("График ROC-кривой Random Forrest")
        plt.legend(loc="lower right")
        plt.grid(True)

        plot_path = "roc_curve.png"
        plt.savefig(plot_path)
        plt.close()

        # Логирование модели
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print("График ROC-кривой Random Forrest успехно сохранен")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="RandomForestClassifier",
            serialization_format="pickle",
        )


# Универсальная функция запуска других моделей


def ml_run_uni(X_train, X_test, y_train, y_test, model):
    numeric_features = [
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "Avg_exp_5",
        "Avg_exp_4",
        "Avg_exp_3",
        "Avg_exp_2",
        "Avg_exp_1",
    ]
    categorical_features = [
        "PAY_1",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "SE_MA",
        "AgeBin",
        "SE_AG",
        "Client_6",
        "Client_5",
        "Client_4",
        "Client_3",
        "Client_2",
        "Client_1",
    ]

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("credit-scoring-model")
    pipeline = create_pipeline(numeric_features, categorical_features, model=model)
    with mlflow.start_run(run_name="credit-scoring-model"):
        mlflow.log_param("model_type", model)
        mlflow.log_params(pipeline["classifier"].get_params())
        print(f"Запуск pipeline для {model}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        # Подсчет метрик

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Метрика accuracy для {model}: {accuracy:.4f}")
        print(f"Метрика precision для {model}: {precision:.4f}")
        print(f"Метрика recall для {model}: {recall:.4f}")
        print(f"Метрика f1-score для {model}: {f1:.4f}")
        print(f"Метрика ROC-AUC для {model}: {auc:.4f}")
        result_dict[model] = round(auc, 4)
        # Логирование метрик

        mlflow.log_metric(f"accuracy {model}", accuracy)
        mlflow.log_metric(f"precision {model}", precision)
        mlflow.log_metric(f"recall {model}", recall)
        mlflow.log_metric(f"f1 {model}", f1)
        mlflow.log_metric(f"auc {model}", auc)

        # Построение графика ROC-кривой

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        plt.figure(figsize=(16, 10))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC кривая (area = %0.2f)" % auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"График ROC-кривой {model}")
        plt.legend(loc="lower right")
        plt.grid(True)

        plot_path = f"roc_curve.png"
        plt.savefig(plot_path)
        plt.close()

        # Логирование модели
        mlflow.log_artifact(plot_path, artifact_path="plots")
        print(f"График ROC-кривой {model} успешно сохранен")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=model,
            serialization_format="pickle",
        )


def copy_pkl(model):
    try:
        path_mlflow = mlflow.artifacts.download_artifacts(f"models:/{model}/latest")
        shutil.copy(next(Path(path_mlflow).rglob("model.pkl")), f"{path}models/model.pkl")
    except Exception as e:
        print(e)
        sys.exit(1)


def save_metrics(best_model_name, metrics_dict, hyperparams=None):
    try:
        metrics_file = Path(f"{path}metrics") / "metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            "model_name": best_model_name,
            "train_test_split": {"test_size": 0.2, "random_state": 42},
            "metrics": metrics_dict,
            "best_roc_auc": result_dict[best_model_name],
        }

        if hyperparams:
            metrics_data["hyperparameters"] = hyperparams

        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f" Метрики успешно сохранены в {metrics_file}")
    except Exception as e:
        print(f" Ошибка при сохранении метрик: {e}")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    mlfow_run_rand_forrest(X_train, X_test, y_train, y_test)
    ml_run_uni(X_train, X_test, y_train, y_test, "logistic_regression")
    ml_run_uni(X_train, X_test, y_train, y_test, "gradient_boosting")
    ml_run_uni(X_train, X_test, y_train, y_test, "xgboost")
    ml_run_uni(X_train, X_test, y_train, y_test, "catboost")
    best_result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)[0]
    print(
        f"Лучший результат у модели {best_result[0]} со значением ROC-AUC {result_dict[best_result[0]]} "
    )
    print(f"Копирование файла модели {best_result[0]}")
    copy_pkl(best_result[0])
    print(f"Копирование файла модели успешно завершено")

    # Сохранить метрики лучшей модели для DVC
    metrics_dict = {"roc_auc": result_dict[best_result[0]]}
    save_metrics(best_result[0], metrics_dict)


if __name__ == "__main__":
    main()
