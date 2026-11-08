import pandas as pd

# колонку PAY_1 в датасете где-то потеряли
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # история просрочек
    out["avg_delay"] = out[PAY_COLS].mean(axis=1)
    out["max_delay"] = out[PAY_COLS].max(axis=1)
    out["num_late_months"] = (out[PAY_COLS] > 0).sum(axis=1)

    # финансовая нагрузка
    out["sum_bill"] = out[BILL_COLS].sum(axis=1)
    out["sum_pay"] = out[PAY_AMT_COLS].sum(axis=1)
    out["utilization"] = out["sum_bill"] / (out["LIMIT_BAL"] + 1e-3)

    # делаем возраст дискретным
    out["age_group"] = pd.cut(
        out["AGE"],
        bins=[17, 24, 34, 44, 54, 64, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype(str)

    return out


NUMERIC_FEATS = [
    "LIMIT_BAL", "AGE", "avg_delay", "max_delay",
    "num_late_months", "sum_bill", "sum_pay", "utilization",
    "BILL_AMT1", "PAY_AMT1",
]

CATEGORICAL_FEATS = ["SEX", "EDUCATION", "MARRIAGE", "age_group"]
