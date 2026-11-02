import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_RAW = os.path.join("data", "raw", "UCI_Credit_Card.csv")
DATA_OUT = os.path.join("data", "processed")
TARGET_COL = "default.payment.next.month"


def grab_data():
    df = pd.read_csv(DATA_RAW)
    df.drop(columns=["ID"], errors="ignore", inplace=True)
    return df


def tidy_up(df):
    # убираем дубли
    df = df.drop_duplicates()
    df = df[(df.AGE.between(18, 95))]
    # PAY_* могут быть -2 (нет истории), это ок, оставим
    return df.reset_index(drop=True)


def split_and_save(df, test_frac=0.2, seed=42):
    os.makedirs(DATA_OUT, exist_ok=True)
    tr, te = train_test_split(df, test_size=test_frac, random_state=seed, stratify=df[TARGET_COL])
    tr.to_csv(os.path.join(DATA_OUT, "train.csv"), index=False)
    te.to_csv(os.path.join(DATA_OUT, "test.csv"), index=False)
    print(f"saved: train={len(tr)}, test={len(te)}")


if __name__ == "__main__":
    raw = grab_data()
    print(f"loaded rows: {len(raw)}")
    cleaned = tidy_up(raw)
    print(f"after cleaning: {len(cleaned)}")
    split_and_save(cleaned)