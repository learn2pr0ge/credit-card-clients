# src/features

Фичеринг — создание новых признаков из очищенных данных.

## Структура

```
features/
└── build_features.py   # Делает новые признаки для модели
```

## Описание

### build_features.py

Назначение: собрать новые признаки поверх очищенного датасета.

Какие признаки делает:

- **SE_MA** — комбинация SEX и MARRIAGE
- **AgeBin** — бины возраста (20-30, 30-40, 40-50, 50-60, 60-70, 70-81)
- **SE_AG** — комбинация SEX и AgeBin
- **Client_1..6** — был ли клиент активен в нужный месяц (по PAY/BILL_AMT/PAY_AMT)
- **Avg_exp_1..5** — средние расходы клиента относительно лимита

Входные данные: `data/raw/credit-scoring-model-cleaned.csv`

Выходные данные: `data/expectations/credit-scoring-model-featured.csv`

## Запуск

```bash
python src/features/build_features.py
```
