"""
MRM-Bench pilot - Clean Logistic Regression baseline
=====================================================
Task   : P(loan reaches 90+ days past due in months t+1..t+12 | information at or before t)
Data   : Freddie Mac SFLLD 2015 sample, Release 47 (July 2026 file layout)
Design : annual December snapshots; train 2015-2018, test 2021-2024 (pandemic years held out);
         loan-disjoint split (no loan ID appears in both train and test).

Model (written out so the benchmark can test against it)
--------------------------------------------------------
  z_i = b0 + x_i' b
  p_i = sigmoid(z_i) = 1 / (1 + exp(-z_i))
  Fitted by minimizing  -sum_i [ y_i log p_i + (1-y_i) log(1-p_i) ] + (1/(2C)) ||b||^2
  (scikit-learn LogisticRegression, L2 penalty, intercept not penalized, lbfgs solver).
  No class weighting, so predicted probabilities stay on the true-event-rate scale (needed for Brier/calibration).

Outcome rules (draft spec, pending professor approval)
------------------------------------------------------
  At risk at t : has a record at t, no zero-balance code at or before t, status at t in {00,01,02}.
  Event   (1)  : any month in t+1..t+12 with numeric status >= 03 or status 'RA'.
  Non-event(0) : no event and the loan is still observed at t+12.
  Censored     : no event and the loan leaves before t+12 (prepay, other termination, end of data)
                 -> excluded from fitting, but counted and reported.

HOW TO RUN
  VS Code / terminal :  python baseline_logreg.py --data /path/to/folder_or_zip
  Google Colab       :  upload sample_2015.zip, then in a cell:
                        !python baseline_logreg.py --data /content/sample_2015.zip
  The folder (or zip) must contain sample_orig_2015.txt and sample_perf_2015.txt.
  Outputs go to ./outputs/ (metrics.json, coefficients.csv, calibration.csv/png, label_counts.csv).
"""
import argparse, hashlib, json, platform, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261022
TRAIN_SNAPSHOTS = [201512, 201612, 201712, 201812]   # windows end by Dec 2019 (pre-pandemic)
TEST_SNAPSHOTS = [202112, 202212, 202312, 202412]    # windows end by Dec 2025
STRESS_SNAPSHOTS = [201912, 202012]                  # pandemic windows: reported only, never used to fit
HORIZON = 12
TRAIN_LOAN_SHARE = 70                                # % of loans (by hash) eligible for training

# ---------------------------------------------------------------- 1. column layout (July 2026)
ORIG_COLS = ["fico", "first_payment_date", "first_time_homebuyer", "maturity_date", "msa", "mi_pct",
             "num_units", "occupancy_status", "orig_cltv", "orig_dti", "orig_upb", "orig_ltv",
             "orig_interest_rate", "channel", "ppm_flag", "amortization_type", "property_state",
             "property_type", "postal_code", "loan_id", "loan_purpose", "orig_loan_term",
             "num_borrowers", "seller_name", "super_conforming_flag", "pre_harp_loan_id",
             "special_eligibility_program", "harp_indicator", "property_valuation_method",
             "io_indicator", "vantagescore4"]
PERF_COLS = ["loan_id", "period", "current_upb", "delinq_status", "loan_age", "rem_months_to_maturity",
             "defect_settlement_date", "modification_flag", "zero_balance_code", "zero_balance_date",
             "current_interest_rate", "current_non_int_upb", "ddlpi", "mi_recoveries",
             "net_sales_proceeds", "non_mi_recoveries", "total_expenses", "legal_costs",
             "maint_pres_costs", "taxes_insurance", "misc_expenses", "actual_loss", "cum_mod_costs",
             "step_mod_flag", "payment_deferral_flag", "eltv", "zb_removal_upb",
             "delinq_accrued_interest", "delinq_due_to_disaster", "borrower_assistance_plan",
             "current_period_mod_costs", "current_int_upb", "mi_cancellation_flag", "servicer_name",
             "bankruptcy_cramdown_costs"]
assert len(ORIG_COLS) == 31 and len(PERF_COLS) == 35

# Documented "Not Available" codes (General User Guide, July 2026) -> converted to NaN
SENTINELS = {"fico": "9999", "mi_pct": "999", "num_units": "99", "orig_cltv": "999",
             "orig_dti": "999", "orig_ltv": "999", "num_borrowers": "99"}


def ym_index(s):
    """YYYYMM -> months since year 0 (so t+12 is simple integer arithmetic)."""
    s = pd.Series(s).astype(int)
    return (s // 100) * 12 + (s % 100) - 1


# ---------------------------------------------------------------- 2. load
def open_files(data):
    data = Path(data)
    if data.suffix == ".zip":
        out = Path("sflld_extracted"); out.mkdir(exist_ok=True)
        with zipfile.ZipFile(data) as z:
            z.extractall(out)
        data = out
    orig = next(data.rglob("sample_orig_*.txt")); perf = next(data.rglob("sample_perf_*.txt"))
    return orig, perf


def load(data):
    of, pf = open_files(data)
    orig = pd.read_csv(of, sep="|", header=None, names=ORIG_COLS, dtype=str,
                       keep_default_na=False, na_values=[""])
    usecols = ["loan_id", "period", "current_upb", "delinq_status", "loan_age", "modification_flag",
               "zero_balance_code", "delinq_due_to_disaster", "borrower_assistance_plan"]
    perf = pd.read_csv(pf, sep="|", header=None, names=PERF_COLS, dtype=str, usecols=usecols,
                       keep_default_na=False, na_values=[""])
    # structural checks against the documentation
    assert orig.loan_id.is_unique, "duplicate loan IDs in origination file"
    assert set(perf.loan_id) <= set(orig.loan_id), "performance loans missing from origination"
    assert not perf.duplicated(["loan_id", "period"]).any(), "duplicate loan-month records"
    perf["t"] = ym_index(perf.period).values
    dq = pd.to_numeric(perf.delinq_status, errors="coerce")     # 'RA' -> NaN here
    perf["dq"] = dq
    perf["sev"] = (dq >= 3) | (perf.delinq_status == "RA")       # 90+ or REO acquisition
    perf["current_upb"] = pd.to_numeric(perf.current_upb)
    perf["loan_age"] = pd.to_numeric(perf.loan_age)
    print(f"Loaded {len(orig):,} loans, {len(perf):,} monthly records, "
          f"periods {perf.period.min()}-{perf.period.max()}")
    return orig, perf


# ---------------------------------------------------------------- 3. origination features
def origination_features(orig):
    o = orig.copy()
    for c, code in SENTINELS.items():
        o.loc[o[c] == code, c] = np.nan
    num = ["fico", "mi_pct", "num_units", "orig_cltv", "orig_dti", "orig_ltv",
           "orig_interest_rate", "orig_upb", "orig_loan_term", "num_borrowers"]
    for c in num:
        o[c] = pd.to_numeric(o[c])
    o["dti_missing"] = o.orig_dti.isna().astype(int)   # almost all are HARP loans (guide: HARP DTI = N/A)
    o["log_orig_upb"] = np.log(o.orig_upb)
    o["term_360"] = (o.orig_loan_term == 360).astype(int)
    o["multi_borrower"] = (o.num_borrowers >= 2).astype(int)
    o["mi_flag"] = (o.mi_pct > 0).astype(int)
    keep = ["loan_id", "fico", "orig_ltv", "orig_dti", "dti_missing", "orig_interest_rate",
            "log_orig_upb", "term_360", "multi_borrower", "mi_flag", "num_units",
            "loan_purpose", "occupancy_status", "first_time_homebuyer", "property_type", "channel"]
    return o[keep]


# ---------------------------------------------------------------- 4. snapshot: features at t + label from t+1..t+12
def build_snapshot(perf, T):
    t = int(ym_index([T])[0])
    hist = perf[perf.t <= t]                                   # ONLY information at or before t
    now = hist[hist.t == t]
    now = now[now.zero_balance_code.isna() & ~now.sev]         # at-risk population
    ids = now.loan_id

    h12 = hist[(hist.t > t - 12) & hist.loan_id.isin(ids)]
    g12 = h12.groupby("loan_id")
    ever = hist[hist.loan_id.isin(ids)].groupby("loan_id")
    feat = now.set_index("loan_id")[["delinq_status", "loan_age", "current_upb",
                                     "delinq_due_to_disaster", "borrower_assistance_plan"]].copy()
    feat["status_t"] = feat.pop("delinq_status")               # '00', '01', '02'
    feat["max_dq_12m"] = g12.dq.max().clip(upper=3)            # 0..3 (3 = was 90+ in last 12m)
    feat["n_dq_months_12m"] = (h12.dq >= 1).groupby(h12.loan_id).sum()
    feat["ever_90_before_t"] = ever.sev.any().astype(int)
    hh = hist[hist.loan_id.isin(ids)]
    feat["ever_modified"] = hh.modification_flag.notna().groupby(hh.loan_id).any().astype(int)
    feat["disaster_flag_t"] = feat.pop("delinq_due_to_disaster").eq("Y").astype(int)
    feat["assist_plan_t"] = feat.pop("borrower_assistance_plan").fillna("none")
    feat["max_feature_period"] = ever.t.max()                  # used by the leakage assertion
    assert (feat.max_feature_period <= t).all(), "LEAKAGE: feature uses a record after t"

    # ---- label (future window, never used as a feature)
    win = perf[(perf.t > t) & (perf.t <= t + HORIZON) & perf.loan_id.isin(ids)]
    event = win.groupby("loan_id").sev.any().reindex(ids, fill_value=False)
    last_t = perf[perf.loan_id.isin(ids)].groupby("loan_id").t.max().reindex(ids)
    zb = (perf[perf.loan_id.isin(ids) & perf.zero_balance_code.notna()]
          .set_index("loan_id").zero_balance_code.reindex(ids))
    label = np.where(event.values, "event",
             np.where(last_t.values >= t + HORIZON, "non_event",
              np.where(zb.values == "01", "censored_prepay",
               np.where(pd.notna(zb.values), "censored_other_termination", "censored_end_of_data"))))
    feat["label"] = label
    feat["snapshot"] = T
    return feat.reset_index()


def loan_pool(loan_id):
    """Deterministic, reproducible loan-level split (no randomness to seed)."""
    return int(hashlib.sha256(loan_id.encode()).hexdigest(), 16) % 100 < TRAIN_LOAN_SHARE


# ---------------------------------------------------------------- 5. metrics
def cluster_bootstrap_ci(df, metric, n=1000, seed=SEED):
    """95% CI resampling LOANS (a loan can appear in several test snapshots)."""
    rng = np.random.default_rng(seed)
    loans = df.loan_id.unique()
    groups = {k: v.index.values for k, v in df.groupby("loan_id")}
    stats = []
    for _ in range(n):
        pick = rng.choice(loans, size=len(loans), replace=True)
        idx = np.concatenate([groups[l] for l in pick])
        y, p = df.y.values[idx], df.p.values[idx]
        if y.min() == y.max():
            continue
        stats.append(metric(y, p))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def evaluate(df, name):
    y, p = df.y.values, df.p.values
    res = {"n": int(len(y)), "events": int(y.sum()), "event_rate": float(y.mean()),
           "mean_predicted": float(p.mean()),
           "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
           "brier": float(brier_score_loss(y, p)),
           "brier_reference_base_rate": float(brier_score_loss(y, np.full_like(p, y.mean())))}
    res["roc_auc_95ci"] = cluster_bootstrap_ci(df, roc_auc_score)
    res["pr_auc_95ci"] = cluster_bootstrap_ci(df, average_precision_score)
    print(f"\n[{name}] n={res['n']:,} events={res['events']} rate={res['event_rate']:.4f} "
          f"mean_p={res['mean_predicted']:.4f}")
    print(f"  ROC-AUC {res['roc_auc']:.3f} {np.round(res['roc_auc_95ci'], 3)} | "
          f"PR-AUC {res['pr_auc']:.3f} {np.round(res['pr_auc_95ci'], 3)} | "
          f"Brier {res['brier']:.5f} (base-rate model {res['brier_reference_base_rate']:.5f})")
    return res


def calibration_table(df, bins=10):
    d = df.assign(bin=pd.qcut(df.p.rank(method="first"), bins, labels=False))
    return d.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                observed_rate=("y", "mean"), events=("y", "sum")).reset_index()


# ---------------------------------------------------------------- 6. main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder or sample_2015.zip")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    np.random.seed(SEED)

    orig, perf = load(args.data)
    ofeat = origination_features(orig)

    snaps = pd.concat([build_snapshot(perf, T) for T in TRAIN_SNAPSHOTS + STRESS_SNAPSHOTS + TEST_SNAPSHOTS])
    snaps = snaps.merge(ofeat, on="loan_id", how="left", validate="many_to_one")
    snaps["in_train_pool"] = snaps.loan_id.map(loan_pool)

    counts = snaps.groupby(["snapshot", "label"]).size().unstack(fill_value=0)
    counts.to_csv(out / "label_counts.csv")
    print("\nLabel counts by snapshot (all loans, before the loan split):\n", counts)

    uncensored = snaps[snaps.label.isin(["event", "non_event"])].copy()
    uncensored["y"] = (uncensored.label == "event").astype(int)
    train = uncensored[uncensored.snapshot.isin(TRAIN_SNAPSHOTS) & uncensored.in_train_pool]
    test = uncensored[uncensored.snapshot.isin(TEST_SNAPSHOTS) & ~uncensored.in_train_pool]
    stress = uncensored[uncensored.snapshot.isin(STRESS_SNAPSHOTS) & ~uncensored.in_train_pool]
    overlap = set(train.loan_id) & set(test.loan_id)
    assert not overlap, f"{len(overlap)} loans in both train and test"
    assert train.snapshot.max() < test.snapshot.min()
    print(f"\nTrain rows {len(train):,} (events {train.y.sum()}) | Test rows {len(test):,} "
          f"(events {test.y.sum()}) | loan overlap = 0")

    numeric = ["fico", "orig_ltv", "orig_dti", "dti_missing", "orig_interest_rate", "log_orig_upb",
               "term_360", "multi_borrower", "mi_flag", "num_units", "loan_age", "max_dq_12m",
               "n_dq_months_12m", "ever_90_before_t", "ever_modified", "disaster_flag_t"]
    categorical = ["status_t", "loan_purpose", "occupancy_status", "first_time_homebuyer",
                   "property_type", "channel", "assist_plan_t"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical)])
    model = Pipeline([("pre", pre),
                      ("lr", LogisticRegression(C=1.0, max_iter=5000, random_state=SEED))])
    model.fit(train[numeric + categorical], train.y)          # preprocessing fitted on TRAIN only

    results = {}
    for name, d in [("train", train), ("test_2021_2024", test), ("stress_2019_2020", stress)]:
        d = d.copy(); d["p"] = model.predict_proba(d[numeric + categorical])[:, 1]
        d = d.reset_index(drop=True)
        results[name] = evaluate(d, name)
        if name == "test_2021_2024":
            cal = calibration_table(d); cal.to_csv(out / "calibration.csv", index=False)
            print("\nCalibration (test, deciles of predicted risk):\n", cal.round(4).to_string(index=False))
            try:
                import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4.5, 4.5))
                ax.plot(cal.mean_predicted, cal.observed_rate, "o-", color="black", label="model")
                m = max(cal.mean_predicted.max(), cal.observed_rate.max()) * 1.05
                ax.plot([0, m], [0, m], "--", color="grey", label="perfect")
                ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed 90+ rate")
                ax.set_title("Calibration, test 2021-2024"); ax.legend(frameon=False)
                fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=150)
            except ImportError:
                pass

    names = model.named_steps["pre"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef_standardized": model.named_steps["lr"].coef_[0]})
    coefs.loc[len(coefs)] = ["intercept", model.named_steps["lr"].intercept_[0]]
    coefs.to_csv(out / "coefficients.csv", index=False)
    print("\nCoefficients (numeric features standardized):\n",
          coefs.sort_values("coef_standardized", key=abs, ascending=False).round(3).to_string(index=False))

    results["config"] = {"seed": SEED, "train_snapshots": TRAIN_SNAPSHOTS, "test_snapshots": TEST_SNAPSHOTS,
                         "horizon_months": HORIZON, "train_loan_share_pct": TRAIN_LOAN_SHARE,
                         "C": 1.0, "features_numeric": numeric, "features_categorical": categorical}
    results["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
