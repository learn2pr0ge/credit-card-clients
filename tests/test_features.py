import pandas as pd

from features.build_features import make_features


class TestFeatures:
    ## Тесты для проверки создания новых признаков

    def test_make_features_returns_dataframe(self, sample_cleaned_data):
        ## Тест проверяет что функция возвращает DataFrame
        result = make_features(sample_cleaned_data.copy())
        assert isinstance(result, pd.DataFrame)

    def test_make_features_creates_new_columns(self, sample_cleaned_data):
        ## Тест проверяет что создаются новие столбцы (признаки)
        df = sample_cleaned_data.copy()
        original_cols_count = len(df.columns)
        result = make_features(df)

        ## Должно быть добавлено несколько новых столбцов
        assert len(result.columns) > original_cols_count

        ## Проверяем что созданы ключевые признаки
        new_features = ["SE_MA", "AgeBin", "SE_AG", "Client_1", "Avg_exp_1"]
        for feature in new_features:
            assert feature in result.columns

    def test_make_features_no_null_values(self, sample_cleaned_data):
        ## Тест проверяет что в нових признаках нет NaN значений
        df = sample_cleaned_data.copy()
        result = make_features(df)

        ## Новые признаки не должны содержать NaN
        new_features = [
            "SE_MA",
            "AgeBin",
            "SE_AG",
            "Client_1",
            "Client_2",
            "Avg_exp_1",
            "Avg_exp_5",
        ]

        for col in new_features:
            assert result[col].isna().sum() == 0
