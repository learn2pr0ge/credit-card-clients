from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def create_pipeline(numeric_features, categorical_features, model="random_forest"):

    model_dict = {
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=42),
        "catboost": CatBoostClassifier(verbose=False, random_state=42),
    }
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model_dict[model])])
    return pipeline
