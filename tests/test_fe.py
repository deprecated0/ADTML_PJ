from src.fe import engineer_features


def test_new_columns_appear(mini_df):
    result = engineer_features(mini_df)
    expected_cols = {
        "avg_delay",
        "max_delay",
        "num_late_months",
        "sum_bill",
        "sum_pay",
        "utilization",
        "age_group",
    }
    assert expected_cols.issubset(set(result.columns))


def test_row_count_unchanged(mini_df):
    assert len(engineer_features(mini_df)) == len(mini_df)


def test_utilization_positive(mini_df):
    result = engineer_features(mini_df)
    assert (result["utilization"] >= 0).all()
