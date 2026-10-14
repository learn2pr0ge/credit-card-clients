from pathlib import Path
import sys

import great_expectations as gx
import pandas as pd

path = Path(__file__).resolve().parent
path = str(path)
path = path[:-8]


# Функция загрузки датасета
def load_df():
    try:
        df = pd.read_csv(f"{path}data/expectations/credit-scoring-model-featured.csv")
        return df
    except FileNotFoundError:
        print("Файл датасета не найден")
        return sys.exit(1)


# Функция GreatExpectations проверки данных
def validate(df: pd.DataFrame):
    context = gx.get_context()
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    # Проверка на допустимые значения данных
    expectations = [
        gx.expectations.ExpectColumnValuesToBeBetween(column="AGE", max_value=110, min_value=1),
        gx.expectations.ExpectColumnValuesToBeInSet(column="target", value_set=[0, 1]),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="LIMIT_BAL", max_value=10000000, min_value=5000
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="LIMIT_BAL", max_value=10000000, min_value=5000
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(column="SEX", value_set=[1, 2]),
        gx.expectations.ExpectColumnValuesToBeInSet(column="MARRIAGE", value_set=[1, 2, 3]),
        gx.expectations.ExpectColumnValuesToBeInSet(column="EDUCATION", value_set=[1, 2, 3, 4]),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="PAY_1", value_set=[0, 1, 2, 3, 4, 5, 6, 7, 8]
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="PAY_2", value_set=[0, 1, 2, 3, 4, 5, 6, 7, 8]
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="PAY_3", value_set=[0, 1, 2, 3, 4, 5, 6, 7, 8]
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="PAY_4", value_set=[0, 1, 2, 3, 4, 5, 6, 7, 8]
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="PAY_5", value_set=[0, 1, 2, 3, 4, 5, 6, 7, 8]
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="PAY_6", value_set=[0, 1, 2, 3, 4, 5, 6, 7, 8]
        ),
        # Проверки на тип поля
        gx.expectations.ExpectColumnValuesToBeOfType(column="LIMIT_BAL", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="SEX", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="EDUCATION", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="MARRIAGE", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="AGE", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_1", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_2", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_3", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_4", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_5", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_6", type_="int64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="BILL_AMT1", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="BILL_AMT2", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="BILL_AMT3", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="BILL_AMT4", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="BILL_AMT5", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="BILL_AMT6", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_AMT1", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_AMT2", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_AMT3", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_AMT4", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_AMT5", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="PAY_AMT6", type_="float64"),
        gx.expectations.ExpectColumnValuesToBeOfType(column="target", type_="int64"),
    ]

    val_results = []
    for exp in expectations:
        validation_results = batch.validate(exp)
        val_results.append(validation_results)

    # Цикл проверки на Null

    for cols in df.columns.tolist():
        exp = gx.expectations.ExpectColumnValuesToNotBeNull(column=cols)
        validation_results = batch.validate(exp)
        val_results.append(validation_results)

    return val_results


# Тест вывода ошибок. Возвращает True/False в зависимости от того проходят данные проверку или нет
def validation_test(validation_list) -> bool:
    true_list = set([d["success"] for d in validation_list])
    if False in true_list:
        return False
    else:
        return True


def save_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.to_csv(f"{path}data/processed/credit-scoring-model-ready_for_model.csv", index=False)
        return 1
    except Exception as e:
        print(e)
        sys.exit(1)


def main():
    df = load_df()
    validation_list = validate(df)
    test_result = validation_test(validation_list)
    if test_result:
        print("Тесты данных прошли успешно")
        print("Сохраняем финальный датасет")
        save_df(df)
        return True
    else:
        print("Данные содержат ошибки")
        sys.exit(1)


if __name__ == "__main__":
    main()
