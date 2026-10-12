class TestDataValidation:
    ## Тесты для проверки валидации данных

    def test_target_is_binary(self, sample_cleaned_data):
        ## Тест проверяет что target имеет только значения 0 или 1
        assert set(sample_cleaned_data["target"].unique()).issubset({0, 1})

    def test_categorical_values_valid(self, sample_cleaned_data):
        ## Тест проверяет что категориальные признаки имеют допустимые значения
        ## Проверяем SEX
        assert set(sample_cleaned_data["SEX"].unique()).issubset({1, 2})

        ## Проверяем EDUCATION
        assert set(sample_cleaned_data["EDUCATION"].unique()).issubset({1, 2, 3, 4})

        ## Проверяем MARRIAGE
        assert set(sample_cleaned_data["MARRIAGE"].unique()).issubset({1, 2, 3})

    def test_pay_columns_valid(self, sample_cleaned_data):
        ## Тест проверяет что PAY_* столбцы имеют допустимые значения (0-3)
        pay_cols = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        valid_pay_values = {0, 1, 2, 3}

        for col in pay_cols:
            assert set(sample_cleaned_data[col].unique()).issubset(valid_pay_values)

    def test_no_nulls_in_critical_columns(self, sample_cleaned_data):
        ## Тест проверяет отсутствие пропусков в критичних столбцах
        critical_cols = ["target", "AGE", "SEX", "EDUCATION", "MARRIAGE", "LIMIT_BAL"]

        for col in critical_cols:
            assert sample_cleaned_data[col].isna().sum() == 0, f"Найдены NaN в {col}"
