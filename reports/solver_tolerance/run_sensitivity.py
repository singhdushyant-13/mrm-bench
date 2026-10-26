"""Solver-tolerance sensitivity of the MRM-Bench baseline logistic regression.

A read-only study. It runs the code cells of notebooks/baseline_logistic_regression.ipynb through section 8 in a fresh
namespace, so the data, labels, features, chronological split, loan partition, preprocessing and C are exactly the
notebook's. It then refits the SAME pipeline changing only the solver setting:
  - lbfgs tol=1e-4: scikit-learn's default, used by the archived first reference (reference/archive/v1_lbfgs_tol_1e-4/);
  - lbfgs tol=1e-6;
  - lbfgs tol=1e-8: the official baseline since 25 October 2026 (reference/);
  - newton-cholesky: an independent solver of the same objective, as a check of the exact optimum.
Metrics, CIs and the calibration table come from the notebook's own evaluate() and calibration code. The tol=1e-8 fit
must reproduce reference/ and the tol=1e-4 fit must reproduce the v1 archive, both within 1e-9.

It writes aggregates only (no loan IDs or loan-level rows) to this folder, and never writes to the notebook,
reference/ or outputs/.

Run from the repository root (about 1 minute):
    uv run python reports/solver_tolerance/run_sensitivity.py
"""
import contextlib
import io
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")      # the notebook's plot cells run without a display

import nbformat
import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.linear_model._logistic as sk_logistic
from scipy import optimize as scipy_optimize
from scipy.special import expit
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)                                   # the notebook reads data/ and reference/ relative to the root
NOTEBOOK = ROOT / "notebooks" / "baseline_logistic_regression.ipynb"
ARCHIVE_V1 = ROOT / "reference" / "archive" / "v1_lbfgs_tol_1e-4"

FITS = [                                         # (label, parameters changed on the notebook's pipeline)
    ("tol=1e-4 (archived v1)", {"lr__tol": 1e-4}),
    ("tol=1e-6", {"lr__tol": 1e-6}),
    ("tol=1e-8 (official)", {"lr__tol": 1e-8}),
    ("newton-cholesky (check)", {"lr__solver": "newton-cholesky", "lr__tol": 1e-10}),
]
PREVIOUS, OFFICIAL, CHECK = FITS[0][0], FITS[2][0], FITS[3][0]
SETS = {"train": "train", "test_2021_2024": "test", "stress_2019_2020": "stress"}   # metrics.json name -> variable


# ---------------------------------------------------------------- 1. run the notebook (sections 1-8) read-only
def find_cell(cells, text):
    """The single code cell whose source contains `text`."""
    hits = [i for i, c in enumerate(cells) if c.cell_type == "code" and text in c.source]
    assert len(hits) == 1, f"expected one cell containing {text!r}, found {hits}"
    return hits[0]


nb = nbformat.read(NOTEBOOK, 4)
fit_cell = find_cell(nb.cells, "model.fit(X_train, y_train)")
setup_cells = [i for i in range(fit_cell + 1) if nb.cells[i].cell_type == "code"]
definition_cells = [find_cell(nb.cells, "def cluster_bootstrap_ci("), find_cell(nb.cells, "def evaluate("),
                    find_cell(nb.cells, "def add_predictions(")]
calibration_source = nb.cells[find_cell(nb.cells, "calibration_data = scored_test.copy()")].source

ns = {"__name__": "__notebook__"}
notebook_log = io.StringIO()


def run(source):
    """Execute notebook code in the shared namespace, with its printed output and tables kept out of the report."""
    with contextlib.redirect_stdout(notebook_log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(source, "<notebook>", "exec"), ns)
    ns["display"] = lambda *args, **kwargs: None          # replaces IPython's display() once it is imported
    if "plt" in ns:
        ns["plt"].close("all")


print(f"running notebook code cells 0..{fit_cell} and {definition_cells} (output suppressed) ...")
for i in setup_cells + definition_cells:
    run(nb.cells[i].source)

notebook_model = ns["model"]                                  # the notebook's fitted official model
X_train, y_train = ns["X_train"], ns["y_train"]
C = notebook_model.named_steps["lr"].C
assert notebook_model.named_steps["lr"].solver == "lbfgs" and notebook_model.named_steps["lr"].tol == 1e-8


# ---------------------------------------------------------------- 2. refit, recording scipy's termination result
class RecordingOptimize:
    """Stands in for scipy.optimize inside sklearn's _logistic module and keeps the last minimize() result."""
    last = None

    def minimize(self, *args, **kwargs):
        RecordingOptimize.last = scipy_optimize.minimize(*args, **kwargs)
        return RecordingOptimize.last

    def __getattr__(self, name):
        return getattr(scipy_optimize, name)


def design_matrix(pipeline, data):
    """The preprocessed inputs (39 columns) as a dense array."""
    Xt = pipeline.named_steps["pre"].transform(data)
    return Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)


def objective(w, b, Xt, y):
    """sklearn's lbfgs objective f = mean log-loss + ||w||^2 / (2 C n) (intercept unpenalized), the documented
    primal J = C * sum(log-loss) + ||w||^2 / 2 = C n f, and the largest absolute gradient component of f."""
    n = len(y)
    z = Xt @ w + b
    log_loss_sum = float(np.sum(np.logaddexp(0.0, z) - y * z))
    penalty = 0.5 * float(w @ w)
    residual = expit(z) - y
    gradient = np.append(Xt.T @ residual / n + w / (C * n), residual.mean())
    return {"f": log_loss_sum / n + penalty / (C * n), "J": C * log_loss_sum + penalty,
            "log_loss_sum": log_loss_sum, "penalty_half_w2": penalty, "max_abs_gradient_f": float(np.abs(gradient).max())}


Xt_train = design_matrix(notebook_model, X_train)
y_arr = y_train.to_numpy(dtype=float)
feature_names = list(notebook_model.named_steps["pre"].get_feature_names_out())

fits, models = {}, {}
original_optimize = sk_logistic.optimize
sk_logistic.optimize = RecordingOptimize()
try:
    for label, params in FITS:
        RecordingOptimize.last = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = clone(notebook_model).set_params(**params).fit(X_train, y_train)
        lr = model.named_steps["lr"]
        assert np.array_equal(design_matrix(model, X_train), Xt_train)     # same preprocessing as the notebook
        res = RecordingOptimize.last
        fits[label] = {
            "solver": lr.solver, "tol": lr.tol, "max_iter": lr.max_iter, "n_iter": int(lr.n_iter_[0]),
            "convergence_warnings": [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)],
            "other_warnings": sorted({f"{w.category.__name__}: {w.message}" for w in caught
                                      if not issubclass(w.category, ConvergenceWarning)}),
            "scipy_status": None if res is None else int(res.status),
            "scipy_message": None if res is None else str(res.message),
            "scipy_max_abs_jac": None if res is None else float(np.abs(res.jac).max()),
            **objective(lr.coef_[0], lr.intercept_[0], Xt_train, y_arr),
        }
        models[label] = model
        print(f"{label:26s} iterations {fits[label]['n_iter']:5d}  status {fits[label]['scipy_status']}  "
              f"warnings {len(fits[label]['convergence_warnings'])}  max|grad f| {fits[label]['max_abs_gradient_f']:.2e}")
finally:
    sk_logistic.optimize = original_optimize


# ---------------------------------------------------------------- 3. coefficients and why some move a lot
coef = pd.DataFrame({label: np.append(m.named_steps["lr"].coef_[0], m.named_steps["lr"].intercept_[0])
                     for label, m in models.items()}, index=feature_names + ["intercept"])
coef.index.name = "feature"

groups = {f: [j for j, name in enumerate(feature_names) if name.startswith(f"cat__{f}_")]
          for f in ns["categorical_features"]}
assert sum(len(cols) for cols in groups.values()) == sum(name.startswith("cat__") for name in feature_names)
design = {set_name: design_matrix(notebook_model, ns[var][ns["numeric_features"] + ns["categorical_features"]])
          for set_name, var in SETS.items()}

# (a) every training row has exactly one active level per categorical group, so each group's columns add up to
#     the intercept's column of ones: shifting a group by d and the intercept by -d leaves the data loss unchanged.
one_active = {f: int((Xt_train[:, cols].sum(axis=1) == 1).sum()) for f, cols in groups.items()}
unseen = {s: {f: int((Xs[:, cols].sum(axis=1) == 0).sum()) for f, cols in groups.items()} for s, Xs in design.items()}
w_prev = models[PREVIOUS].named_steps["lr"].coef_[0].copy()
b_prev = models[PREVIOUS].named_steps["lr"].intercept_[0]
w_shift = w_prev.copy()
w_shift[groups["assist_plan_t"]] += 0.5
shift_check = {"data_log_loss_change": objective(w_shift, b_prev - 0.5, Xt_train, y_arr)["log_loss_sum"]
               - objective(w_prev, b_prev, Xt_train, y_arr)["log_loss_sum"],
               "penalty_change": 0.5 * float(w_shift @ w_shift - w_prev @ w_prev)}

# (b) along that direction only the penalty acts, and it is smallest when the group's coefficients sum to zero.
group_sums = pd.DataFrame({label: {f: float(coef[label].iloc[cols].sum()) for f, cols in groups.items()}
                           for label in coef.columns})


def centred(column):
    """Coefficients with each group's mean removed and added to the intercept: the part that sets predictions."""
    out = column.copy()
    for cols in groups.values():
        mean = column.iloc[cols].mean()
        out.iloc[cols] -= mean
        out.loc["intercept"] += mean
    return out


coef_centred = coef.apply(centred)
level_counts = pd.DataFrame({"train_rows": Xt_train.sum(axis=0), "train_events": Xt_train.T @ y_arr},
                            index=feature_names).astype(int)

decomposition = []
for f, cols in groups.items():
    delta = coef[OFFICIAL].iloc[cols] - coef[PREVIOUS].iloc[cols]
    for j in cols:
        name = feature_names[j]
        decomposition.append({
            "group": f, "feature": name, "train_rows": int(level_counts.loc[name, "train_rows"]),
            "train_events": int(level_counts.loc[name, "train_events"]),
            "coef_v1_tol_1e-4": coef.loc[name, PREVIOUS], "coef_official_tol_1e-8": coef.loc[name, OFFICIAL],
            "change": coef.loc[name, OFFICIAL] - coef.loc[name, PREVIOUS],
            "common_group_shift": float(delta.mean()),
            "contrast_change": coef_centred.loc[name, OFFICIAL] - coef_centred.loc[name, PREVIOUS]})
decomposition = pd.DataFrame(decomposition)


def coef_summary(a, b):
    d = (coef[b] - coef[a]).abs()
    dc = (coef_centred[b] - coef_centred[a]).abs()
    num = [n.startswith("num__") for n in coef.index]
    cat = [n.startswith("cat__") for n in coef.index]
    return {"max_abs_all": float(d.max()), "argmax": d.idxmax(), "max_abs_numeric": float(d[num].max()),
            "argmax_numeric": d[num].idxmax(), "max_abs_categorical": float(d[cat].max()),
            "abs_intercept": float(d["intercept"]), "max_abs_centred_categorical": float(dc[cat].max()),
            "argmax_centred": dc[cat].idxmax(), "abs_centred_intercept": float(dc["intercept"])}


# (c) split v1's objective gap: centring its groups changes no prediction, only the penalty.
w_c, b_c = coef_centred[PREVIOUS].iloc[:-1].to_numpy(), coef_centred[PREVIOUS].iloc[-1]
centring_check = {
    "J_v1": fits[PREVIOUS]["J"], "J_v1_centred": objective(w_c, b_c, Xt_train, y_arr)["J"],
    "J_optimum_newton": fits[CHECK]["J"],
    "max_abs_logit_change_from_centring": float(np.abs((Xt_train @ w_c + b_c) - (Xt_train @ w_prev + b_prev)).max())}

coef_pairs = {f"{a} -> {b}": coef_summary(a, b) for a, b in
              [(PREVIOUS, "tol=1e-6"), (PREVIOUS, OFFICIAL), ("tol=1e-6", OFFICIAL), (OFFICIAL, CHECK),
               (PREVIOUS, CHECK)]}


# ---------------------------------------------------------------- 4. predictions, metrics and calibration
def recalibration(y, p):
    """Calibration intercept and slope: logistic regression of y on logit(p), fitted by Newton's method with
    step halving (p clipped away from 0 and 1 so the logit stays finite)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    X = np.column_stack([np.ones_like(p), np.log(p / (1 - p))])

    def neg_log_lik(beta):
        z = X @ beta
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    beta = np.array([0.0, 1.0])
    for _ in range(100):
        q = expit(X @ beta)
        hessian = X.T @ (X * (q * (1 - q))[:, None]) + 1e-12 * np.eye(2)
        step = np.linalg.solve(hessian, X.T @ (y - q))
        size = 1.0
        while neg_log_lik(beta + size * step) > neg_log_lik(beta) and size > 1e-8:
            size /= 2
        beta = beta + size * step
        if np.abs(size * step).max() < 1e-10:
            break
    return {"calibration_intercept": float(beta[0]), "calibration_slope": float(beta[1])}


metrics, scored, calibration = {}, {}, {}
for label, model in models.items():
    print(f"evaluating {label} (1,000 loan-bootstrap resamples per metric) ...")
    ns["model"] = model                                   # add_predictions() scores with the namespace's model
    scored[label], metrics[label] = {}, {}
    for set_name, var in SETS.items():
        with contextlib.redirect_stdout(notebook_log):
            scored[label][set_name] = ns["add_predictions"](ns[var])
            metrics[label][set_name] = ns["evaluate"](scored[label][set_name], set_name)
        s = scored[label][set_name]
        metrics[label][set_name].update(recalibration(s["y"].to_numpy(float), s["p"].to_numpy()))
    ns["scored_test"] = scored[label]["test_2021_2024"]
    run(calibration_source)
    calibration[label] = ns["calibration"].copy()
ns["model"] = notebook_model


def max_diff_from_files(label, folder):
    """Largest absolute difference between a fit's metrics/CIs/coefficients/calibration and the files in folder
    (the same items and the same 1e-9 tolerance as notebook section 12)."""
    ref_metrics = json.loads((folder / "metrics.json").read_text())
    ref_coef = pd.read_csv(folder / "coefficients.csv").set_index("feature")["coef_standardized"]
    ref_cal = pd.read_csv(folder / "calibration.csv")
    diffs = [abs(metrics[label][s][k] - ref_metrics[s][k]) for s in SETS for k in ref_metrics[s]
             if not k.endswith("_95ci")]
    diffs += [abs(a - b) for s in SETS for k in ref_metrics[s] if k.endswith("_95ci")
              for a, b in zip(metrics[label][s][k], ref_metrics[s][k])]
    diffs += list((coef[label] - ref_coef.reindex(coef.index)).abs())
    diffs += list(np.abs(calibration[label][ref_cal.columns].to_numpy(float) - ref_cal.to_numpy(float)).ravel())
    return float(max(diffs))


reproduction = {"official tol=1e-8 vs reference/": max_diff_from_files(OFFICIAL, ROOT / "reference"),
                "tol=1e-4 vs reference/archive/v1_lbfgs_tol_1e-4/": max_diff_from_files(PREVIOUS, ARCHIVE_V1)}
for what, d in reproduction.items():
    print(f"{what}: max difference {d:.1e} -> {'reproduced' if d <= 1e-9 else 'NOT REPRODUCED'}")

prob_diffs = []
for label in models:
    for base in [PREVIOUS, OFFICIAL]:
        if label == base:
            continue
        for set_name in SETS:
            p0, p1 = scored[base][set_name]["p"], scored[label][set_name]["p"]
            d = (p1 - p0).abs()
            row = {"fit": label, "compared_with": base, "set": set_name, "max_abs": float(d.max()),
                   "mean_abs": float(d.mean()), "p99_abs": float(d.quantile(0.99)),
                   "max_relative": float((d / p0).max()), "spearman": float(p0.corr(p1, method="spearman"))}
            if set_name == "test_2021_2024":
                bins = lambda p: pd.qcut(p.rank(method="first"), 10, labels=False)
                row["test_rows_changing_decile"] = int((bins(p0) != bins(p1)).sum())
            prob_diffs.append(row)
prob_diffs = pd.DataFrame(prob_diffs)

calibration_summary = {}
for label, table in calibration.items():
    gap = (table["observed_rate"] - table["mean_predicted"]).abs()
    calibration_summary[label] = {"max_decile_gap": float(gap.max()),
                                  "weighted_mean_decile_gap": float((gap * table["n"]).sum() / table["n"].sum()),
                                  "top_decile_mean_predicted": float(table["mean_predicted"].iloc[-1]),
                                  "top_decile_observed_rate": float(table["observed_rate"].iloc[-1])}


# ---------------------------------------------------------------- 5. rare category levels: v1 -> official
# A level is "rare" when fewer than 1,000 training rows have it. For each set and level: subgroup calibration
# (observed rate with a Wilson 95% interval vs mean prediction under each fit), Brier, size of the change, and
# membership of the set's top risk decile. Then the level contrasts (the interpretable part of one-hot coefficients).
rare_levels = [n for n in feature_names if n.startswith("cat__") and level_counts.loc[n, "train_rows"] < 1000]


def wilson(k, n, z=1.959964):
    """Wilson 95% interval for an observed proportion k / n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


subgroups, by_level, largest = [], [], []
for set_name in SETS:
    Xs = design[set_name]
    y = scored[OFFICIAL][set_name]["y"].to_numpy(float)
    p_prev = scored[PREVIOUS][set_name]["p"].to_numpy()
    p_off = scored[OFFICIAL][set_name]["p"].to_numpy()
    d = np.abs(p_off - p_prev)
    top_prev = p_prev >= np.quantile(p_prev, 0.9)                    # the set's top 10% of predicted risk
    top_off = p_off >= np.quantile(p_off, 0.9)
    rare_row = Xs[:, [feature_names.index(n) for n in rare_levels]].any(axis=1)
    row_groups = [("all rows", np.ones(len(y), bool)), ("rows with a rare level", rare_row),
                  ("all other rows", ~rare_row)] + [(n, Xs[:, feature_names.index(n)] == 1) for n in rare_levels]
    for name, mask in row_groups:
        n_rows, k = int(mask.sum()), int(y[mask].sum())
        if n_rows == 0:
            continue
        low, high = wilson(k, n_rows)
        mp_prev, mp_off = float(p_prev[mask].mean()), float(p_off[mask].mean())
        subgroups.append({
            "set": set_name, "rows": name, "n": n_rows, "events": k, "observed_rate": k / n_rows,
            "observed_95ci_low": low, "observed_95ci_high": high,
            "mean_pred_v1": mp_prev, "mean_pred_official": mp_off,
            "v1_inside_ci": bool(low <= mp_prev <= high), "official_inside_ci": bool(low <= mp_off <= high),
            "brier_v1": float(np.mean((p_prev[mask] - y[mask]) ** 2)),
            "brier_official": float(np.mean((p_off[mask] - y[mask]) ** 2)),
            "mean_abs_dp": float(d[mask].mean()), "max_abs_dp": float(d[mask].max()),
            "share_p_up": float((p_off[mask] > p_prev[mask]).mean()),
            "top_decile_v1": int(top_prev[mask].sum()), "top_decile_official": int(top_off[mask].sum()),
            "moved_into_top_decile": int((top_off & ~top_prev)[mask].sum()),
            "moved_out_of_top_decile": int((top_prev & ~top_off)[mask].sum())})
        if name in ("rows with a rare level", "all other rows") or name in rare_levels:
            by_level.append({"set": set_name, "rows": name, "n": n_rows, "mean_abs_dp": float(d[mask].mean()),
                             "max_abs_dp": float(d[mask].max())})
    # which combinations of rare levels carry the 25 largest changes (aggregate counts, no row identifiers)
    if set_name != "train":
        order = np.argsort(-d)[:25]
        combo = np.array(["+".join(n.replace("cat__", "") for n in rare_levels if Xs[i, feature_names.index(n)] == 1)
                          or "(no rare level)" for i in order])
        for c in pd.unique(combo):
            sel = order[combo == c]
            largest.append({"set": set_name, "rare levels of the row": c, "rows among the 25 largest changes": len(sel),
                            "mean_p_v1": float(p_prev[sel].mean()), "mean_p_official": float(p_off[sel].mean()),
                            "max_abs_dp": float(d[sel].max()), "events": int(y[sel].sum())})
subgroups, by_level, largest = pd.DataFrame(subgroups), pd.DataFrame(by_level), pd.DataFrame(largest)

contrasts = []
for f, cols in groups.items():
    names = [feature_names[j] for j in cols]
    base = max(names, key=lambda n: level_counts.loc[n, "train_rows"])      # compare with the most common level
    for n in names:
        if n == base:
            continue
        lo_prev = coef.loc[n, PREVIOUS] - coef.loc[base, PREVIOUS]
        lo_off = coef.loc[n, OFFICIAL] - coef.loc[base, OFFICIAL]
        contrasts.append({"feature": f, "level": n.replace(f"cat__{f}_", ""), "versus": base.replace(f"cat__{f}_", ""),
                          "train_rows": int(level_counts.loc[n, "train_rows"]),
                          "train_events": int(level_counts.loc[n, "train_events"]),
                          "odds_ratio_v1": float(np.exp(lo_prev)), "odds_ratio_official": float(np.exp(lo_off)),
                          "odds_ratio_check_newton": float(np.exp(coef.loc[n, CHECK] - coef.loc[base, CHECK])),
                          "direction_changed": bool((lo_prev > 0) != (lo_off > 0))})
contrasts = pd.DataFrame(contrasts)


# ---------------------------------------------------------------- 6. write the outputs (aggregates only)
coef.join(coef_centred.add_suffix(" [centred]")).to_csv(HERE / "coefficients_by_fit.csv")
decomposition.to_csv(HERE / "categorical_coefficient_changes.csv", index=False)
pd.concat({label: t for label, t in calibration.items()}, names=["fit"]).reset_index(level=0).to_csv(
    HERE / "calibration_test_by_fit.csv", index=False)
prob_diffs.to_csv(HERE / "probability_differences.csv", index=False)
by_level.to_csv(HERE / "probability_differences_by_level.csv", index=False)
subgroups.to_csv(HERE / "rare_level_subgroups.csv", index=False)
largest.to_csv(HERE / "largest_changes_composition.csv", index=False)
contrasts.to_csv(HERE / "level_contrasts.csv", index=False)
pd.DataFrame([{"fit": label, "set": s, **{k: v for k, v in m.items() if not k.endswith("_95ci")},
               "roc_auc_95ci": m["roc_auc_95ci"], "pr_auc_95ci": m["pr_auc_95ci"]}
              for label, by_set in metrics.items() for s, m in by_set.items()]).to_csv(HERE / "metrics_by_fit.csv",
                                                                                       index=False)
results = {
    "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                 "scipy": scipy.__version__},
    "setup": {"C": C, "n_train": int(len(y_arr)), "train_events": int(y_arr.sum()), "n_parameters": len(coef),
              "penalty_weight_in_f_1_over_Cn": 1 / (C * len(y_arr)), "reproduction_max_differences": reproduction},
    "fits": fits, "coefficient_differences": coef_pairs, "group_coefficient_sums": group_sums.to_dict(),
    "training_rows_with_exactly_one_active_level": one_active, "rows_with_unseen_category": unseen,
    "group_shift_check_assist_plan_t_plus_0.5": shift_check, "objective_gap_split": centring_check,
    "rare_levels_under_1000_train_rows": rare_levels, "metrics": metrics,
    "calibration_summary": calibration_summary,
}
(HERE / "results.json").write_text(json.dumps(results, indent=1, default=float))


def md_table(df, floatfmt="{:.4g}"):
    cells = [[floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in df.itertuples(index=False)]
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(r) + " |\n" for r in cells)


labels = list(models)
lines = ["# Solver-tolerance sensitivity: generated tables", "",
         "Generated by `run_sensitivity.py` (do not edit by hand; see `README.md` for the interpretation).", "",
         f"scikit-learn {sklearn.__version__}, scipy {scipy.__version__}. Same data, labels, features, splits, loan "
         f"partition, preprocessing and C = {C} as the notebook; only the solver setting changes.", "",
         "Reproduction checks (largest absolute difference over every metric, CI, coefficient and calibration value; "
         "must be ≤ 1e-9):", ""]
lines += [f"- {what}: {d:.1e}" for what, d in reproduction.items()]
lines += ["", "## 1. Termination and objective (training data, n = "
          f"{len(y_arr):,}; f = mean log-loss + ||w||²/(2Cn), J = C·Σ log-loss + ||w||²/2 = C·n·f)", ""]
lines.append(md_table(pd.DataFrame([{
    "fit": label, "iterations": v["n_iter"], "scipy status": v["scipy_status"], "scipy message": v["scipy_message"],
    "ConvergenceWarnings": len(v["convergence_warnings"]), "max abs grad f": f"{v['max_abs_gradient_f']:.2e}",
    "J": f"{v['J']:.6f}", "J − J(newton)": f"{v['J'] - fits[CHECK]['J']:.2e}"} for label, v in fits.items()])))
lines += ["", "## 2. Coefficient differences (39 coefficients + intercept)", "",
          "'centred' = each categorical group's mean moved into the intercept, the part that determines predictions.", ""]
lines.append(md_table(pd.DataFrame([{"comparison": k, "max abs (all)": v["max_abs_all"], "where": v["argmax"],
                                     "max abs numeric": v["max_abs_numeric"], "where (numeric)": v["argmax_numeric"],
                                     "max abs categorical": v["max_abs_categorical"],
                                     "abs intercept": v["abs_intercept"], "max abs centred categorical":
                                     v["max_abs_centred_categorical"], "where (centred)": v["argmax_centred"]}
                                    for k, v in coef_pairs.items()])))
lines += ["", "### Sum of each categorical group's coefficients (0 at the exact optimum)", ""]
lines.append(md_table(group_sums.reset_index().rename(columns={"index": "group"})))
lines += ["", "### Categorical coefficient changes, v1 (tol=1e-4) → official (tol=1e-8), with training rows and events", ""]
lines.append(md_table(decomposition.round(4)))
lines += ["", f"Shift check: adding 0.5 to every assist_plan_t coefficient and −0.5 to the intercept changes the "
          f"training log-loss sum by {shift_check['data_log_loss_change']:.1e} and ||w||²/2 by "
          f"{shift_check['penalty_change']:.3f}. Training rows with exactly one active level per group: "
          f"{one_active} of {len(y_arr):,}. Rows with an unseen category (all zeros in a group): {unseen}.", "",
          f"Objective gap split: J(v1) = {centring_check['J_v1']:.4f}; the same fit with each group centred (no "
          f"prediction changes: max logit change {centring_check['max_abs_logit_change_from_centring']:.1e}) has J = "
          f"{centring_check['J_v1_centred']:.4f}; the optimum (newton-cholesky) has J = "
          f"{centring_check['J_optimum_newton']:.4f}.", ""]
lines += ["## 3. Metrics (point estimates; 95% loan-bootstrap CIs as in the notebook)", ""]
rows = []
for label in labels:
    for s in SETS:
        m = metrics[label][s]
        rows.append({"fit": label, "set": s, "mean pred": f"{m['mean_predicted']:.4%}", "event rate":
                     f"{m['event_rate']:.4%}", "ROC-AUC": f"{m['roc_auc']:.4f} [{m['roc_auc_95ci'][0]:.3f}, "
                     f"{m['roc_auc_95ci'][1]:.3f}]", "PR-AUC": f"{m['pr_auc']:.4f} [{m['pr_auc_95ci'][0]:.3f}, "
                     f"{m['pr_auc_95ci'][1]:.3f}]", "Brier": f"{m['brier']:.6f}", "base-rate Brier":
                     f"{m['brier_reference_base_rate']:.6f}", "cal. intercept": f"{m['calibration_intercept']:.3f}",
                     "cal. slope": f"{m['calibration_slope']:.3f}"})
lines.append(md_table(pd.DataFrame(rows)))
lines += ["", "## 4. Probability differences", ""]
lines.append(md_table(prob_diffs))
lines += ["", "### v1 → official probability changes by row type (rare = a level with < 1,000 training rows)", ""]
lines.append(md_table(by_level))
lines += ["", "## 5. Test calibration by decile of predicted risk", ""]
cal_wide = pd.DataFrame({"bin": calibration[OFFICIAL]["bin"], "n": calibration[OFFICIAL]["n"]})
for label in labels:
    cal_wide[f"pred {label}"] = calibration[label]["mean_predicted"].values
    cal_wide[f"obs {label}"] = calibration[label]["observed_rate"].values
lines.append(md_table(cal_wide, "{:.5f}"))
lines.append(md_table(pd.DataFrame([{"fit": k, **v} for k, v in calibration_summary.items()]), "{:.5f}"))
lines += ["", "## 6. Rare category levels: subgroup calibration, Brier and top-decile membership (v1 → official)", "",
          "Observed rate with a Wilson 95% interval; 'inside' = the subgroup's mean prediction lies inside that interval. "
          "Top decile = the set's 10% highest predictions under each fit.", ""]
sub_view = subgroups.copy()
for col in ["observed_rate", "observed_95ci_low", "observed_95ci_high", "mean_pred_v1", "mean_pred_official"]:
    sub_view[col] = sub_view[col].map(lambda v: f"{v:.2%}")
lines.append(md_table(sub_view, "{:.4g}"))
lines += ["", "### Rare-level combinations among the 25 largest probability changes", ""]
lines.append(md_table(largest, "{:.4f}"))
lines += ["", "### Level contrasts: odds ratio versus the feature's most common level", ""]
lines.append(md_table(contrasts, "{:.3f}"))
(HERE / "report.md").write_text("\n".join(lines).rstrip("\n") + "\n")      # exactly one newline at the end
print("wrote", ", ".join(p.name for p in sorted(HERE.glob("*")) if p.suffix in {".md", ".csv", ".json"}))
