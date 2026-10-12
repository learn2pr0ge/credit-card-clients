from pathlib import Path
import sys

import pandas as pd

path = Path(__file__).resolve().parent
path = str(path)
path = path[:-12]


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(f"{path}data/raw/credit-scoring-model-cleaned.csv")
        return df
    except FileNotFoundError:
        print("Файл датасета не найден")
        return sys.exit(1)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Делаем новые feature по SEX и Marriage
    df["SE_MA"] = 0
    df.loc[((df.SEX == 1) & (df.MARRIAGE == 1)), "SE_MA"] = 1
    df.loc[((df.SEX == 1) & (df.MARRIAGE == 2)), "SE_MA"] = 2
    df.loc[((df.SEX == 1) & (df.MARRIAGE == 3)), "SE_MA"] = 3
    df.loc[((df.SEX == 2) & (df.MARRIAGE == 1)), "SE_MA"] = 4
    df.loc[((df.SEX == 2) & (df.MARRIAGE == 2)), "SE_MA"] = 5
    df.loc[((df.SEX == 2) & (df.MARRIAGE == 3)), "SE_MA"] = 6
    # Делаем бины возраста
    df["AgeBin"] = 0
    df.loc[((df["AGE"] > 20) & (df["AGE"] < 30)), "AgeBin"] = 1
    df.loc[((df["AGE"] >= 30) & (df["AGE"] < 40)), "AgeBin"] = 2
    df.loc[((df["AGE"] >= 40) & (df["AGE"] < 50)), "AgeBin"] = 3
    df.loc[((df["AGE"] >= 50) & (df["AGE"] < 60)), "AgeBin"] = 4
    df.loc[((df["AGE"] >= 60) & (df["AGE"] < 70)), "AgeBin"] = 5
    df.loc[((df["AGE"] >= 70) & (df["AGE"] < 81)), "AgeBin"] = 6
    # Соединяем sex/agebin
    df["SE_AG"] = 0
    df.loc[((df.SEX == 1) & (df.AgeBin == 1)), "SE_AG"] = 1
    df.loc[((df.SEX == 1) & (df.AgeBin == 2)), "SE_AG"] = 2
    df.loc[((df.SEX == 1) & (df.AgeBin == 3)), "SE_AG"] = 3
    df.loc[((df.SEX == 1) & (df.AgeBin == 4)), "SE_AG"] = 4
    df.loc[((df.SEX == 1) & (df.AgeBin == 5)), "SE_AG"] = 5
    df.loc[((df.SEX == 2) & (df.AgeBin == 1)), "SE_AG"] = 6
    df.loc[((df.SEX == 2) & (df.AgeBin == 2)), "SE_AG"] = 7
    df.loc[((df.SEX == 2) & (df.AgeBin == 3)), "SE_AG"] = 8
    df.loc[((df.SEX == 2) & (df.AgeBin == 4)), "SE_AG"] = 9
    df.loc[((df.SEX == 2) & (df.AgeBin == 5)), "SE_AG"] = 10
    # Был ли клиент в нужный месяц
    df["Client_6"] = 1
    df["Client_5"] = 1
    df["Client_4"] = 1
    df["Client_3"] = 1
    df["Client_2"] = 1
    df["Client_1"] = 1
    df.loc[((df.PAY_6 == 0) & (df.BILL_AMT6 == 0) & (df.PAY_AMT6 == 0)), "Client_6"] = 0
    df.loc[((df.PAY_5 == 0) & (df.BILL_AMT5 == 0) & (df.PAY_AMT5 == 0)), "Client_5"] = 0
    df.loc[((df.PAY_4 == 0) & (df.BILL_AMT4 == 0) & (df.PAY_AMT4 == 0)), "Client_4"] = 0
    df.loc[((df.PAY_3 == 0) & (df.BILL_AMT3 == 0) & (df.PAY_AMT3 == 0)), "Client_3"] = 0
    df.loc[((df.PAY_2 == 0) & (df.BILL_AMT2 == 0) & (df.PAY_AMT2 == 0)), "Client_2"] = 0
    df.loc[((df.PAY_1 == 0) & (df.BILL_AMT1 == 0) & (df.PAY_AMT1 == 0)), "Client_1"] = 0
    # Feature средних расходов клиента
    df["Avg_exp_5"] = ((df["BILL_AMT5"] - (df["BILL_AMT6"] - df["PAY_AMT5"]))) / df["LIMIT_BAL"]
    df["Avg_exp_4"] = (
        (
            (df["BILL_AMT5"] - (df["BILL_AMT6"] - df["PAY_AMT5"]))
            + (df["BILL_AMT4"] - (df["BILL_AMT5"] - df["PAY_AMT4"]))
        )
        / 2
    ) / df["LIMIT_BAL"]
    df["Avg_exp_3"] = (
        (
            (df["BILL_AMT5"] - (df["BILL_AMT6"] - df["PAY_AMT5"]))
            + (df["BILL_AMT4"] - (df["BILL_AMT5"] - df["PAY_AMT4"]))
            + (df["BILL_AMT3"] - (df["BILL_AMT4"] - df["PAY_AMT3"]))
        )
        / 3
    ) / df["LIMIT_BAL"]
    df["Avg_exp_2"] = (
        (
            (df["BILL_AMT5"] - (df["BILL_AMT6"] - df["PAY_AMT5"]))
            + (df["BILL_AMT4"] - (df["BILL_AMT5"] - df["PAY_AMT4"]))
            + (df["BILL_AMT3"] - (df["BILL_AMT4"] - df["PAY_AMT3"]))
            + (df["BILL_AMT2"] - (df["BILL_AMT3"] - df["PAY_AMT2"]))
        )
        / 4
    ) / df["LIMIT_BAL"]
    df["Avg_exp_1"] = (
        (
            (df["BILL_AMT5"] - (df["BILL_AMT6"] - df["PAY_AMT5"]))
            + (df["BILL_AMT4"] - (df["BILL_AMT5"] - df["PAY_AMT4"]))
            + (df["BILL_AMT3"] - (df["BILL_AMT4"] - df["PAY_AMT3"]))
            + (df["BILL_AMT2"] - (df["BILL_AMT3"] - df["PAY_AMT2"]))
            + (df["BILL_AMT1"] - (df["BILL_AMT2"] - df["PAY_AMT1"]))
        )
        / 5
    ) / df["LIMIT_BAL"]
    # Feature близости платежа

    return df


def save_df(df: pd.DataFrame):
    try:
        df.to_csv(f"{path}data/expectations/credit-scoring-model-featured.csv", index=False)
        return 1
    except Exception as e:
        print(e)
        sys.exit(1)


def main():
    df = load_data()
    df = make_features(df)
    save_df(df)

    return 1


if __name__ == "__main__":
    main()
