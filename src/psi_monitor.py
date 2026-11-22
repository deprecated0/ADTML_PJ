import numpy as np
import pandas as pd
import requests

API = "http://127.0.0.1:8000/predict"
N_BINS = 10


def psi(expected, actual, bins=N_BINS):
    _, edges = np.histogram(expected, bins=bins)
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e_pct = e_counts / e_counts.sum() + 1e-7
    a_pct = a_counts / a_counts.sum() + 1e-7

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def main():
    test_df = pd.read_csv("data/processed/test.csv").sample(150, random_state=7)
    train_df = pd.read_csv("data/processed/train.csv")

    cols_needed = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    ] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]

    probs = []
    errors = 0
    for _, r in test_df[cols_needed].iterrows():
        try:
            resp = requests.post(API, json=r.to_dict(), timeout=0.5)
            probs.append(resp.json()["probability"])
        except requests.ConnectionError:
            print("API не отвечает")
            return
        except Exception:
            errors += 1

    if errors:
        print(f"{errors} запросов не прошли")

    probs = np.array(probs)

    p = psi(train_df["default.payment.next.month"].values, probs)
    print(f"PSI (probabilities): {p:.4f}")

    p2 = psi(train_df["LIMIT_BAL"].values, test_df["LIMIT_BAL"].values)
    print(f"PSI (LIMIT_BAL): {p2:.4f}")

    p3 = psi(train_df["AGE"].values, test_df["AGE"].values)
    print(f"PSI (AGE): {p3:.4f}")

    threshold = 0.2
    if max(p, p2, p3) > threshold:
        print(f"Обнаружен дрейф (порог {threshold})")
    else:
        print("Дрейф в пределах нормы.")


if __name__ == "__main__":
    main()
