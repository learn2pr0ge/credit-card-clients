import logging
from pathlib import Path
import pickle
import sys

from detect_drift import detect_drift_chisquare, detect_drift_ks
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)

app = FastAPI()

path = Path(__file__).resolve().parent
path = str(path)
path = path[:-7]


# Загрузка модели
try:
    with open(f"{path}/models/model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError as exc:
    raise RuntimeError("Нет файла модели") from exc


class Client(BaseModel):
    LIMIT_BAL: float = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=1, le=4)
    MARRIAGE: int = Field(ge=1, le=3)
    AGE: int = Field(ge=18, le=110)
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


class Prediction(BaseModel):
    default: int
    probability: float


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


def prepare_features(client: Client) -> pd.DataFrame:
    X = pd.DataFrame([client.model_dump()])
    # Модель и фичеринг работают с PAY_1
    X = X.rename(columns={"PAY_0": "PAY_1"})
    save_client(X)
    X = make_features(X)
    return X


# Сохраняем всех клиентов в DataFrame

clients_df = pd.DataFrame()


def save_client(df: pd.DataFrame):
    global clients_df
    clients_df = pd.concat([clients_df, df], axis=0, ignore_index=True)
    return clients_df


def check_drift(df: pd.DataFrame):
    if len(df) < 10:
        return None
    # Загрузка данных
    try:
        df_orig = pd.read_csv(f"{path}/data/processed/X_train.csv")
    except FileNotFoundError:
        logger.exception("Файл с датасетом не найден")
        return 1
    drift_report = []

    for column in [
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
    ]:
        is_drift, p_val = detect_drift_ks(df_orig[column], df[column])
        if is_drift:
            drift_report.append(f"Дрифт обнаружили в {column}. P-value: {p_val:.4f}")

    for column in [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_1",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]:
        is_drift, p_val = detect_drift_chisquare(df_orig[column], df[column])
        if is_drift:
            drift_report.append(f"Дрифт обнаружили в {column}. P-value: {p_val:.4f}")
    if len(drift_report) > 0:
        logger.warning(f"Обнаружен дрифт данных {drift_report}")
        return 1


# Endpoint /predict
@app.post("/predict", response_model=Prediction)
async def predict(client: Client):
    try:
        X = prepare_features(client)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1])
        try:
            check_drift(clients_df)
        except Exception:
            logger.warning("Проверка дрифта пропущена")
    except Exception as exc:
        logger.exception("Ошибка при получении предсказания")
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {exc}") from exc
    return Prediction(default=pred, probability=round(prob, 4))


# Endpoint /healthcheck
@app.get("/healthcheck")
async def healthcheck():
    return JSONResponse({"status": "ok"}, status_code=200)
