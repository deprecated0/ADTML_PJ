from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(num_cols, cat_cols):
    num_transformer = Pipeline(
        [
            ("fill_na", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        [
            ("fill_na", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", num_transformer, num_cols),
            ("categorical", cat_transformer, cat_cols),
        ]
    )


def make_pipeline(num_cols, cat_cols, model_type="gbc"):
    prep = make_preprocessor(num_cols, cat_cols)

    if model_type == "gbc":
        clf = GradientBoostingClassifier(random_state=7)
    elif model_type == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=7)
    else:
        raise ValueError(f"Передфн невалидный тип модели '{model_type}'")

    return Pipeline([("prep", prep), ("model", clf)])
