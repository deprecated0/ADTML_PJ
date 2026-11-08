import numpy as np

from src.model_builder import make_pipeline
from src.fe import NUMERIC_FEATS, CATEGORICAL_FEATS


def test_pipeline_builds():
    pipe = make_pipeline(NUMERIC_FEATS, CATEGORICAL_FEATS, model_type="logreg")
    assert pipe is not None
    assert len(pipe.steps) == 2


def test_pipeline_predicts(mini_df):
    from src.fe import engineer_features
    df = engineer_features(mini_df)
    feats = NUMERIC_FEATS + CATEGORICAL_FEATS
    X, y = df[feats], df["default.payment.next.month"]

    pipe = make_pipeline(NUMERIC_FEATS, CATEGORICAL_FEATS, model_type="gbc")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})
