# src/api

FastAPI приложение для отдачи предсказаний по обученной модели.

## Структура

```
api/
└── api_model.py   # FastAPI приложение с endpoint /predict и /healthcheck
```

## Описание

### api_model.py

Назначение: поднять API, которое принимает данные клиента и отдает предсказание дефолта.

Что делает:

- При старте грузит модель из `models/model.pkl`
- Pydantic схема **Client** — тело запроса с признаками клиента (с валидацией диапазонов SEX, AGE и т.д.)
- Pydantic схема **Prediction** — ответ (default и probability)
- `prepare_features` — делает фичеринг для одного клиента (как в build_features) и переименовывает PAY_0 в PAY_1
- Обрабатывает ошибки предсказания и отдает 500 с описанием

Endpoints:

- `POST /predict` — принимает признаки клиента, возвращает предсказание и вероятность дефолта
- `GET /healthcheck` — проверка что сервис живой (отдает {"status": "ok"})

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000.0, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 15000.0, "BILL_AMT2": 12000.0, "BILL_AMT3": 10000.0,
    "BILL_AMT4": 8000.0, "BILL_AMT5": 6000.0, "BILL_AMT6": 4000.0,
    "PAY_AMT1": 2000.0, "PAY_AMT2": 1500.0, "PAY_AMT3": 1000.0,
    "PAY_AMT4": 800.0, "PAY_AMT5": 600.0, "PAY_AMT6": 500.0
  }'
```

Пример ответа:

```json
{
  "default": 0,
  "probability": 0.1234
}
```

## Запуск

```bash
# Локально через uvicorn
uvicorn api_model:app --app-dir src/api --host 0.0.0.0 --port 8000

# Или через Makefile
make serve
```

Swagger UI доступен на http://localhost:8000/docs
