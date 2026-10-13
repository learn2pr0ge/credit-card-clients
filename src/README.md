# src

Весь исходный код проекта. Разбит по этапам: подготовка данных, фичеринг, обучение модели и API.

## Структура

```
src/
├── run_pipeline.py     # Прогоняет пайплайн подготовки данных по шагам
├── data/               # Загрузка, очистка и валидация данных
├── features/           # Фичеринг (создание новых признаков)
├── models/             # Sklearn pipeline и обучение моделей
└── api/                # FastAPI приложение с endpoint /predict
```

## Описание

### run_pipeline.py

Назначение: запускает подготовку данных одной командой.

Что делает: по очереди прогоняет скрипты через subprocess. Если любой шаг завершился с
ненулевым кодом — останавливает работу. Порядок:

1. `data/make_dataset.py` — загрузка и очистка
2. `features/build_features.py` — фичеринг
3. `data/validation.py` — валидация через Great Expectations

Входные данные: `data/raw/UCI_Credit_Card.csv`

Выходные данные: `data/processed/credit-scoring-model-ready_for_model.csv`

## Запуск

```bash
python src/run_pipeline.py
```

## Подпапки

Подробности по каждому этапу смотри в README внутри папок:

- [data/](data/README.md) — подготовка и валидация данных
- [features/](features/README.md) — фичеринг
- [models/](models/README.md) — pipeline и обучение
- [api/](api/README.md) — FastAPI приложение
