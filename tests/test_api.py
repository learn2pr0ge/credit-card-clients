import asyncio

import pandas as pd
from pydantic import ValidationError
import pytest

from api.api_model import Client, Prediction, healthcheck, predict, prepare_features


class TestApi:
    ## Тесты для проверки api модели

    def test_client_model_valid(self, valid_client):
        ## Тест проверяет что Client создается из валидного тела
        client = Client(**valid_client)
        assert client.SEX == valid_client["SEX"]
        assert client.AGE == valid_client["AGE"]

    def test_client_invalid_sex(self, valid_client):
        ## Тест проверяет что невалидний SEX вызивает ошибку
        bad = valid_client.copy()
        bad["SEX"] = 5
        with pytest.raises(ValidationError):
            Client(**bad)

    def test_client_invalid_age(self, valid_client):
        ## Тест проверяет что слишком маленький возраст не проходит
        bad = valid_client.copy()
        bad["AGE"] = 10
        with pytest.raises(ValidationError):
            Client(**bad)

    def test_prepare_features_returns_dataframe(self, valid_client):
        ## Тест проверяет что prepare_features возвращает DataFrame
        X = prepare_features(Client(**valid_client))
        assert isinstance(X, pd.DataFrame)

    def test_prepare_features_renames_pay0(self, valid_client):
        ## Тест проверяет что PAY_0 переименовивается в PAY_1
        X = prepare_features(Client(**valid_client))
        assert "PAY_1" in X.columns
        assert "PAY_0" not in X.columns

    def test_prepare_features_creates_new_features(self, valid_client):
        ## Тест проверяет что создаются новие признаки
        X = prepare_features(Client(**valid_client))
        new_features = ["SE_MA", "AgeBin", "SE_AG", "Client_1", "Avg_exp_1"]
        for feature in new_features:
            assert feature in X.columns

    def test_predict_endpoint(self, valid_client):
        ## Тест проверяет что эндпоинт predict отдает корректний ответ
        result = asyncio.run(predict(Client(**valid_client)))
        assert isinstance(result, Prediction)
        assert result.default in (0, 1)
        assert 0 <= result.probability <= 1

    def test_healthcheck_endpoint(self):
        ## Тест проверяет что healthcheck работает
        response = asyncio.run(healthcheck())
        assert response.status_code == 200
