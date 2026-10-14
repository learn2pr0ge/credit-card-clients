import json
from pathlib import Path
import sys

import pandas as pd
import requests

path = Path(__file__).resolve().parent
path = str(path)
path = path[:-7]


# Загрузка датасета
def load_dataset():
    try:
        df = pd.read_csv(f"{path}/data/processed/X_train.csv")
        # Переименуем поле PAY_1 в PAY_0
        df = df.rename(columns={"PAY_1": "PAY_0"})
        print(df.head())
        return df
    except FileNotFoundError:
        print("Файл датасета не найден")
        sys.exit(1)


# Функция посылки запроса
def make_requests(df):
    # Переименуем поле PAY_1 в PAY_0

    for i in range(30):
        row = df.sample(1).iloc[0].to_dict()
        payload = json.dumps(row)
        print(payload)
        req = requests.post(
            "http://127.0.0.1:8000/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        print(req.status_code)
        print(req.json())


def main():
    df = load_dataset()
    make_requests(df)


if __name__ == "__main__":
    main()
