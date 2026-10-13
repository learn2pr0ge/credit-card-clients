# Система кредитного скоринга

Учебный проект по дисциплине MLOps. Модель предсказывает дефолт клиента (вернет кредит или нет)
по датасету UCI Credit Card. Весь путь собран в один пайплайн: загрузка и очистка данных,
фичеринг, валидация через Great Expectations, обучение пяти моделей с логированием в MLflow,
версионирование через DVC и отдача предсказаний через FastAPI в Docker.

## Архитектура проекта

```
credit-scoring-model/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # CI/CD пайплайн (линтеры, валидация данных, тесты)
├── data/                      # Данные (версионируются DVC, в git не лежат)
│   ├── raw/                   # Сырые данные + очищенный датасет
│   ├── expectations/          # Датасет с фичами для проверки Great Expectations
│   └── processed/             # Финальный датасет готовый для модели
├── metrics/
│   └── metrics.json           # Метрики лучшей модели для DVC
├── models/
│   └── model.pkl              # Обученная модель (версионируется DVC)
├── notebooks/
│   └── eda.ipynb              # Разведочный анализ данных (EDA)
├── src/                       # Весь исходный код
│   ├── run_pipeline.py        # Прогоняет пайплайн подготовки данных
│   ├── data/
│   │   ├── make_dataset.py    # Загрузка и очистка данных
│   │   └── validation.py      # Валидация данных через Great Expectations
│   ├── features/
│   │   └── build_features.py  # Фичеринг (новые признаки)
│   ├── models/
│   │   ├── pipeline.py        # Sklearn pipeline (препроцессинг + модель)
│   │   └── train.py           # Обучение моделей, MLflow, выбор лучшей
│   └── api/
│       └── api_model.py       # FastAPI приложение с endpoint /predict
├── tests/                     # Pytest тесты
├── Dockerfile                 # Образ с FastAPI приложением
├── Makefile                   # Удобные команды (data, train, serve, test, lint)
├── dvc.yaml                   # DVC pipeline (стадии prepare и train)
├── requirements.txt           # Python зависимости
├── pyproject.toml             # Конфиг проекта + black/isort
├── setup.cfg                  # Конфиг flake8
├── start.sh                   # Скрипт полного запуска для Linux
└── README.md
```

## Быстрый старт

### Вариант A: start.sh (Linux, одной командой)

Скрипт сам поставит wget и docker если их нет, скачает датасет, соберет и запустит контейнер.

```bash
git clone https://github.com/learn2pr0ge/credit-card-clients/
chmod +x start.sh
./start.sh
```

После запуска API будет доступно на http://localhost (порт 80 проброшен на 8000 контейнера).

### Вариант B: Локально через Makefile

```bash
# Ставим зависимости
make requirements

# Скачиваем датасет в data/raw
make data

# Подготовка данных (make_dataset -> build_features -> validation)
make prepare

# Обучение моделей
make train

# Запуск FastAPI локально
make serve
```

### Вариант C: Docker вручную

```bash
# Сборка образа
make docker-build

# Запуск контейнера на 8000 порту
make docker-run
```

### Вариант D: DVC pipeline

Прогоняет весь пайплайн (подготовка данных + обучение) по стадиям из dvc.yaml.

```bash
dvc repro
```

## Пайплайн обработки данных и обучения

Подготовка данных собрана в `src/run_pipeline.py` и идет по шагам:

1. `src/data/make_dataset.py` — загрузка сырого датасета, очистка (дроп ID, фикс значений
   MARRIAGE и EDUCATION, переименование PAY_0 в PAY_1 и таргета, нормализация PAY_*).
2. `src/features/build_features.py` — фичеринг: новые признаки SE_MA, AgeBin, SE_AG,
   Client_1..6, Avg_exp_1..5.
3. `src/data/validation.py` — валидация через Great Expectations (диапазоны, типы, отсутствие null).

Обучение в `src/models/train.py`: разбивка train/test (test_size=0.2), sklearn pipeline с
препроцессингом, обучение 5 моделей, GridSearch для Random Forest, логирование в MLflow,
выбор лучшей модели по ROC-AUC и сохранение model.pkl + metrics.json.

## Модель и результаты

Обучаются и сравниваются 5 моделей. Лучшая по ROC-AUC сохраняется в `models/model.pkl`.

| Модель | ROC-AUC |
|--------|---------|
| Gradient Boosting | **0.7814** |
| Random Forest | 0.7806 |
| CatBoost | 0.7790 |
| XGBoost | 0.7647 |
| Logistic Regression | 0.7500 |

Итоговая лучшая модель — **Gradient Boosting**:

| Метрика | Значение |
|---------|----------|
| accuracy | 0.8193 |
| precision | 0.6606 |
| recall | 0.3587 |
| f1-score | 0.4650 |
| ROC-AUC | 0.7814 |

ROC-кривая сохраняется как артефакт (`roc_curve.png`) и логируется в MLflow.

## API (FastAPI)

Приложение в `src/api/api_model.py`. Поднимается через uvicorn (`make serve`) или в Docker.

Endpoints:

- `POST /predict` — принимает признаки клиента, возвращает предсказание и вероятность дефолта
- `GET /healthcheck` — проверка что сервис живой
- `GET /docs` — Swagger UI с интерактивной документацией

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

## MLflow

Все эксперименты логируются в MLflow (трекинг в папку `mlruns`). Логируются параметры моделей,
метрики (accuracy, precision, recall, f1, ROC-AUC), ROC-кривая как артефакт и сама обученная модель.

```bash
# Запуск MLflow UI
mlflow ui

# UI будет на http://localhost:5000
```

## DVC

Данные и модель версионируются через DVC, в git они не лежат. Пайплайн описан в `dvc.yaml`
и состоит из двух стадий:

- `prepare` — подготовка данных (make_dataset -> build_features -> validation)
- `train` — обучение моделей и сохранение model.pkl + metrics.json

```bash
# Прогнать весь пайплайн
dvc repro

# Посмотреть статус
dvc status
```

## Тестирование

Тесты на pytest лежат в `tests/`. Покрывают фичеринг, pipeline, валидацию данных и API.

```bash
# Запуск всех тестов
make test

# Запуск с подсчетом покрытия
make test-cov
```

## CI/CD (GitHub Actions)

Пайплайн `.github/workflows/ci-cd.yml` запускается на push и pull request в main. Состоит из jobs:

- **code-quality** — проверка форматирования (black, isort) и линтинг (flake8)
- **data-validation** — скачивание датасета, подготовка данных и валидация через Great Expectations
- **test** — запуск pytest с подсчетом покрытия

## Разработка

Полезные команды из Makefile:

```bash
make format    # Форматирование кода (isort + black)
make lint      # Проверка линтерами (flake8, isort, black)
make clean     # Чистка кэша и артефактов тестов
```

## Требования к окружению

- Python 3.10+
- Зависимости из `requirements.txt` (`make requirements` или `pip install -r requirements.txt`)
- Docker (для запуска контейнера)

## Лицензия

Проект под лицензией MIT. Подробности в файле [LICENSE](LICENSE).
