# src/models

Sklearn pipeline и обучение моделей с логированием в MLflow.

## Структура

```
models/
├── pipeline.py     # Sklearn pipeline (препроцессинг + модель)
├── train.py        # Обучение моделей, MLflow, выбор лучшей
└── roc_curve.png   # ROC-кривая последней обученной модели
```

## Описание

### pipeline.py

Назначение: собрать единый sklearn pipeline с препроцессингом и моделью.

Что делает:

- Числовые признаки — SimpleImputer (median) + StandardScaler
- Категориальные признаки — OneHotEncoder (handle_unknown="ignore")
- Все это через ColumnTransformer
- На выходе Pipeline из препроцессора и выбранной модели

Доступные модели: random_forest, gradient_boosting, logistic_regression, xgboost, catboost.

### train.py

Назначение: обучить модели, залогировать в MLflow и выбрать лучшую.

Что делает:

- Грузит готовый датасет из `data/processed/`
- Разбивает на train/test (test_size=0.2, random_state=42)
- Random Forest обучается через GridSearchCV (подбор гиперпараметров, cv=5)
- Остальные модели обучаются универсальной функцией
- Считает метрики: accuracy, precision, recall, f1, ROC-AUC
- Строит и сохраняет ROC-кривую, логирует параметры, метрики, артефакты и модель в MLflow
- Выбирает лучшую модель по ROC-AUC, копирует ее в `models/model.pkl`
- Сохраняет метрики лучшей модели в `metrics/metrics.json` для DVC

Входные данные: `data/processed/credit-scoring-model-ready_for_model.csv`

Выходные данные:

- `models/model.pkl` — лучшая обученная модель
- `metrics/metrics.json` — метрики лучшей модели
- `roc_curve.png` — ROC-кривая (артефакт MLflow)

## Запуск

```bash
python src/models/train.py
```
