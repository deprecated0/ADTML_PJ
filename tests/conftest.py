import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def mini_df():
    np.random.seed(0)
    n = 30
    return pd.DataFrame({
        "LIMIT_BAL": np.random.randint(20000, 300000, n).astype(float),
        "SEX": np.random.choice([1, 2], n),
        "EDUCATION": np.random.choice([1, 2, 3], n),
        "MARRIAGE": np.random.choice([1, 2, 3], n),
        "AGE": np.random.randint(20, 65, n),
        "PAY_0": np.random.randint(-1, 4, n),
        "PAY_2": np.random.randint(-1, 4, n),
        "PAY_3": np.random.randint(-1, 4, n),
        "PAY_4": np.random.randint(-1, 4, n),
        "PAY_5": np.random.randint(-1, 4, n),
        "PAY_6": np.random.randint(-1, 4, n),
        "BILL_AMT1": np.random.uniform(0, 50000, n),
        "BILL_AMT2": np.random.uniform(0, 50000, n),
        "BILL_AMT3": np.random.uniform(0, 50000, n),
        "BILL_AMT4": np.random.uniform(0, 50000, n),
        "BILL_AMT5": np.random.uniform(0, 50000, n),
        "BILL_AMT6": np.random.uniform(0, 50000, n),
        "PAY_AMT1": np.random.uniform(0, 10000, n),
        "PAY_AMT2": np.random.uniform(0, 10000, n),
        "PAY_AMT3": np.random.uniform(0, 10000, n),
        "PAY_AMT4": np.random.uniform(0, 10000, n),
        "PAY_AMT5": np.random.uniform(0, 10000, n),
        "PAY_AMT6": np.random.uniform(0, 10000, n),
        "default.payment.next.month": np.random.choice([0, 1], n),
    })
