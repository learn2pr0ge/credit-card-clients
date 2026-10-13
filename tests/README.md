# tests

Pytest тесты. Покрывают фичеринг, pipeline, валидацию данных и API.

## Структура

```
tests/
├── conftest.py              # Фикстуры с тестовыми данными
├── test_api.py              # Тесты FastAPI (Client, prepare_features, predict, healthcheck)
├── test_features.py         # Тесты фичеринга (make_features)
├── test_pipeline.py         # Тесты sklearn pipeline (create_pipeline)
└── test_data_validation.py  # Тесты валидации данных
```

## Описание

### conftest.py

Назначение: общие фикстуры для тестов.

Что внутри:

- Добавляет `src` в path чтобы импортировать модули
- `sample_cleaned_data` — очищенные данные (как после make_dataset)
- `sample_featured_data` — данные с признаками (как после build_features)
- `valid_client` — валидное тело клиента для API
- `numeric_features` / `categorical_features` — списки признаков из модели

### test_api.py

Тесты API: что Client создается из валидного тела, невалидный SEX и AGE падают, prepare_features
возвращает DataFrame и переименовывает PAY_0 в PAY_1 и создает новые признаки, что endpoint
predict отдает корректный ответ, а healthcheck возвращает 200.

### test_features.py

Тесты фичеринга: что make_features возвращает DataFrame, создает новые столбцы и в них нет NaN.

### test_pipeline.py

Тесты pipeline: что create_pipeline создается без ошибок, имеет preprocessor и работает со всеми
5 моделями.

### test_data_validation.py

Тесты валидации данных: target бинарный, категориальные признаки в допустимых значениях, PAY_*
в нужном диапазоне, нет null в критичных столбцах.

## Запуск

```bash
# Все тесты
make test

# С покрытием
make test-cov

# Напрямую через pytest
pytest tests
```
