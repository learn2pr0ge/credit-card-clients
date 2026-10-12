from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from models.pipeline import create_pipeline


class TestPipeline:
    ## Тесты для проверки создания sklearn пайплайна

    def test_pipeline_creates_successfully(self, numeric_features, categorical_features):
        ## Тест проверяет что пайплайн создается без ошибок
        pipeline = create_pipeline(numeric_features, categorical_features, "random_forest")
        assert pipeline is not None
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_preprocessor(self, numeric_features, categorical_features):
        ## Тест проверяет что пайплайн имеет preprocessor
        pipeline = create_pipeline(numeric_features, categorical_features, "gradient_boosting")
        assert "preprocessor" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)

    def test_pipeline_with_different_models(self, numeric_features, categorical_features):
        ## Тест проверяет создание пайплайнов с разними моделями
        models = [
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
            "xgboost",
            "catboost",
        ]

        for model in models:
            pipeline = create_pipeline(numeric_features, categorical_features, model)
            assert pipeline is not None
            assert "classifier" in pipeline.named_steps
