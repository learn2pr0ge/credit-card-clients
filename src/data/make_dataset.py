from logging import exception

import pandas as pd
import sys
from pathlib import Path


path = Path(__file__).resolve().parent
path = str(path)
path = path[:-8]

# Загрузка дата сета
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(f"{path}data/raw/UCI_Credit_Card.csv")
        return df
    except FileNotFoundError:
        print('Файл датасета не найден')
        return sys.exit(1)

# Проверка датасета
def clean_data(df):
    print('=' * 60)
    print(df.isnull().sum())
    print('=' * 60)
    print(df.duplicated().sum())
    print('=' * 60)
    # Удалим столбец ID
    df = df.drop(columns=['ID'])
    # Смотрим распределения Marriage и Education
    print(df['MARRIAGE'].value_counts())
    print('=' * 60)
    print(df['EDUCATION'].value_counts())
    print('=' * 60)
    # Переносим 0 в marriage в other
    fil_marriage = (df['MARRIAGE'] == 0)
    df.loc[fil_marriage, 'MARRIAGE'] = 3
    print(df['MARRIAGE'].value_counts())
    # Переносим 5, 6, 0 education в other
    fil_edu = (df['EDUCATION'] == 0) | (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6)
    df.loc[fil_edu, 'EDUCATION'] = 4
    print(df['EDUCATION'].value_counts())
    print('=' * 60)
    # PAY_0 переименуем в PAY_1 и default.payment.next.month в таргет
    df = df.rename(columns={'PAY_0': 'PAY_1',
                            'default.payment.next.month': 'target',})
    # Меняем -1 и -2 в Pay_* на 0 согласно документации к датасету
    fil = (df.PAY_1 == -2) | (df.PAY_1 == -1) | (df.PAY_1 == 0)
    df.loc[fil, 'PAY_1'] = 0
    fil = (df.PAY_2 == -2) | (df.PAY_2 == -1) | (df.PAY_2 == 0)
    df.loc[fil, 'PAY_2'] = 0
    fil = (df.PAY_3 == -2) | (df.PAY_3 == -1) | (df.PAY_3 == 0)
    df.loc[fil, 'PAY_3'] = 0
    fil = (df.PAY_4 == -2) | (df.PAY_4 == -1) | (df.PAY_4 == 0)
    df.loc[fil, 'PAY_4'] = 0
    fil = (df.PAY_5 == -2) | (df.PAY_5 == -1) | (df.PAY_5 == 0)
    df.loc[fil, 'PAY_5'] = 0
    fil = (df.PAY_6 == -2) | (df.PAY_6 == -1) | (df.PAY_6 == 0)
    df.loc[fil, 'PAY_6'] = 0
    return df


def save_data(df):
    try:
        df.to_csv(f"{path}data/raw/credit-scoring-model-cleaned.csv", index=False)
        return 1
    except Exception as e:
        print(e)
        sys.exit(1)


# Основная функция

def main():
    df = load_data()
    df = clean_data(df)
    print(df.info())

    save_data(df)

if __name__ == "__main__":
    main()

















