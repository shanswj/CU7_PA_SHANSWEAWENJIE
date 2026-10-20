"""Creates YOUR OWN data file for the LangganKu PA.  Usage:  python make_my_data.py <last four digits of your IC>"""
import sys

def master_churn():
    import numpy as np, pandas as pd
    rng = np.random.default_rng(20261008)
    N = 1500
    plan = rng.choice(["Basic", "Standard", "Premium"], N, p=[.40, .38, .22])
    payment_method = rng.choice(["Card", "E-wallet", "Bank transfer", "Cash voucher"], N, p=[.34, .30, .22, .14])
    contract = rng.choice(["Monthly", "6-month", "Yearly"], N, p=[.50, .25, .25])
    main_device = rng.choice(["Phone", "TV", "Laptop", "Tablet"], N, p=[.42, .28, .20, .10])
    tenure_months = np.clip(rng.gamma(2.0, 9.0, N), 1, 72).astype(int)
    support_calls = np.clip(rng.poisson(1.3, N), 0, 8)
    monthly_fee_rm = np.round(np.select([plan == "Premium", plan == "Standard"], [54.9, 39.9], 24.9) + rng.normal(0, 3.0, N), 1)
    avg_hours_week = np.round(np.clip(rng.gamma(2.2, 3.2, N) * np.where(plan == "Premium", 1.25, 1.0), 0.2, 40.0), 1)
    logit = 1.7 * (-0.85 - 0.045 * tenure_months + 0.42 * support_calls + 0.020 * (monthly_fee_rm - 40)
                   - 0.075 * avg_hours_week
                   + np.select([contract == "Monthly", contract == "6-month"], [0.95, 0.25], -0.80)
                   + np.select([payment_method == "Cash voucher", payment_method == "Bank transfer"], [0.55, 0.15], 0.0)
                   + np.select([main_device == "TV", main_device == "Tablet"], [-0.35, 0.25], 0.0)
                   + 0.80 * ((support_calls >= 3) & (contract == "Monthly"))
                   + 0.60 * ((tenure_months <= 3))
                   + rng.normal(0, 0.20, N))
    churned = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"subscriber_id": [f"S{i:04d}" for i in range(1, N + 1)],
                         "tenure_months": tenure_months, "support_calls": support_calls,
                         "monthly_fee_rm": monthly_fee_rm, "avg_hours_week": avg_hours_week,
                         "plan": plan, "payment_method": payment_method, "contract": contract, "main_device": main_device,
                         "churned": churned})

MASTER = master_churn
CFG = {'kind': 'cls', 'mode': 'exam', 'id': 'subscriber_id', 'target': 'churned', 'file': 'subscriber_churn.csv', 'brand': 'LangganKu', 'exp': 'LangganKu_Churn', 'reg': 'langganku_churn', 'db': 'langganku', 'n_rows': 1200, 'n_estimators': 100}


# Needs a global MASTER (the scenario's master-data function) and a dict CFG.
import io, os, re, sys, json, shutil, tempfile, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def build_my_data(L4, cfg):
    """The candidate's own copy of the data: a seeded sample of rows and a shuffled column order."""
    df = MASTER()
    rng = np.random.default_rng(int(L4) + 20261007)
    idx = np.sort(rng.choice(len(df), cfg["n_rows"], replace=False))
    df = df.iloc[idx].reset_index(drop=True)
    feats = [c for c in df.columns if c not in (cfg["id"], cfg["target"])]
    order = [feats[i] for i in rng.permutation(len(feats))]
    return df[[cfg["id"]] + order + [cfg["target"]]]




if __name__ == "__main__":
    if len(sys.argv) != 2 or not (len(sys.argv[1]) == 4 and sys.argv[1].isdigit()):
        sys.exit("Usage: python make_my_data.py <last four digits of your IC>   e.g. python make_my_data.py 0731")
    L4 = sys.argv[1]
    mine = build_my_data(L4, CFG)
    mine.to_csv(CFG["file"], index=False)
    print(f"Created {CFG['file']} for IC ending {L4}: {mine.shape[0]} rows x {mine.shape[1]} columns.")
    print("Use THIS file in your notebook. Do not share it: another candidate's file is different.")
