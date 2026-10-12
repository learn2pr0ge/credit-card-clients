from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

## Добавляем src в path чтобы импортировать модули
path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(path / "src"))


@pytest.fixture
def sample_cleaned_data():
    ## Фикстура для создания очищенных данных (после make_dataset.py)
    np.random.seed(42)
    n_rows = 100

    df = pd.DataFrame(
        {
            "LIMIT_BAL": np.random.uniform(50000, 500000, n_rows).astype("float64"),
            "AGE": np.random.randint(25, 80, n_rows).astype("int64"),
            "PAY_1": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_2": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_3": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_4": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_5": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_6": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "BILL_AMT1": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT2": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT3": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT4": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT5": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT6": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "PAY_AMT1": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT2": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT3": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT4": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT5": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT6": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "SEX": np.random.choice([1, 2], n_rows).astype("int64"),
            "EDUCATION": np.random.choice([1, 2, 3, 4], n_rows).astype("int64"),
            "MARRIAGE": np.random.choice([1, 2, 3], n_rows).astype("int64"),
            "target": np.random.choice([0, 1], n_rows).astype("int64"),
        }
    )

    return df


@pytest.fixture
def sample_featured_data():
    ## Фикстура для создания данных с признаками (после build_features.py)
    np.random.seed(42)
    n_rows = 100

    df = pd.DataFrame(
        {
            "LIMIT_BAL": np.random.uniform(50000, 500000, n_rows).astype("float64"),
            "AGE": np.random.randint(25, 80, n_rows).astype("int64"),
            "PAY_1": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_2": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_3": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_4": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_5": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "PAY_6": np.random.choice([0, 1, 2, 3], n_rows).astype("int64"),
            "BILL_AMT1": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT2": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT3": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT4": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT5": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "BILL_AMT6": np.random.uniform(0, 200000, n_rows).astype("float64"),
            "PAY_AMT1": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT2": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT3": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT4": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT5": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "PAY_AMT6": np.random.uniform(0, 100000, n_rows).astype("float64"),
            "SEX": np.random.choice([1, 2], n_rows).astype("int64"),
            "EDUCATION": np.random.choice([1, 2, 3, 4], n_rows).astype("int64"),
            "MARRIAGE": np.random.choice([1, 2, 3], n_rows).astype("int64"),
            "target": np.random.choice([0, 1], n_rows).astype("int64"),
            ## Новые признаки из build_features
            "SE_MA": np.random.randint(0, 7, n_rows).astype("int64"),
            "AgeBin": np.random.randint(0, 7, n_rows).astype("int64"),
            "SE_AG": np.random.randint(0, 11, n_rows).astype("int64"),
            "Client_1": np.random.choice([0, 1], n_rows).astype("int64"),
            "Client_2": np.random.choice([0, 1], n_rows).astype("int64"),
            "Client_3": np.random.choice([0, 1], n_rows).astype("int64"),
            "Client_4": np.random.choice([0, 1], n_rows).astype("int64"),
            "Client_5": np.random.choice([0, 1], n_rows).astype("int64"),
            "Client_6": np.random.choice([0, 1], n_rows).astype("int64"),
            "Avg_exp_1": np.random.uniform(-1, 1, n_rows).astype("float64"),
            "Avg_exp_2": np.random.uniform(-1, 1, n_rows).astype("float64"),
            "Avg_exp_3": np.random.uniform(-1, 1, n_rows).astype("float64"),
            "Avg_exp_4": np.random.uniform(-1, 1, n_rows).astype("float64"),
            "Avg_exp_5": np.random.uniform(-1, 1, n_rows).astype("float64"),
        }
    )

    return df


@pytest.fixture
def numeric_features():
    ## Список числовых признаков из модели
    return [
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


@pytest.fixture
def categorical_features():
    ## Список категориальних признаков из модели
    return [
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
