# src/data

Загрузка, очистка и валидация данных.

## Структура

```
data/
├── make_dataset.py   # Загрузка и очистка сырого датасета
└── validation.py     # Валидация данных через Great Expectations
```

## Описание

### make_dataset.py

Назначение: загрузка сырого датасета и первичная очистка.

Что делает:

- Грузит `data/raw/UCI_Credit_Card.csv`
- Смотрит пропуски и дубликаты
- Дропает столбец ID
- Переносит 0 в MARRIAGE в other (3)
- Переносит 0, 5, 6 в EDUCATION в other (4)
- Переименовывает PAY_0 в PAY_1 и default.payment.next.month в target
- Меняет -1 и -2 в PAY_* на 0 согласно документации к датасету

Входные данные: `data/raw/UCI_Credit_Card.csv`

Выходные данные: `data/raw/credit-scoring-model-cleaned.csv`

### validation.py

Назначение: проверка данных перед обучением через Great Expectations.

Что делает:

- Грузит датасет с фичами из `data/expectations/`
- Проверяет допустимые значения (AGE, target, LIMIT_BAL, SEX, MARRIAGE, EDUCATION, PAY_*)
- Проверяет типы полей (int64 / float64)
- Циклом проверяет отсутствие null во всех столбцах
- Если хоть одна проверка не прошла — падает с ошибкой, иначе сохраняет финальный датасет

Входные данные: `data/expectations/credit-scoring-model-featured.csv`

Выходные данные: `data/processed/credit-scoring-model-ready_for_model.csv`

## Запуск

```bash
# Очистка данных
python src/data/make_dataset.py

# Валидация (запускать после фичеринга)
python src/data/validation.py
```
