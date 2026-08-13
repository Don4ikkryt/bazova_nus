"""Етап 2. Шкалювання за Рашем, психометрика тесту, DIF.

На виході:
  output/students.csv (доповнено колонками score_*, theta_*)
  output/02_psychometrics.json
"""
import json
import re

import numpy as np
import pandas as pd

from common import OUT, DOMAINS, GROUP_ORDER, SUBJECTS, cronbach_alpha, subject_of
import irt

students = pd.read_csv(OUT / "students.csv")
items = pd.read_csv(OUT / "items.csv")
key = pd.read_csv(OUT / "key.csv")

assert (students["id"].values == items["id"].values).all()
item_cols = [c for c in items.columns if c != "id"]
X = items[item_cols].values.astype(float)

report = {"scales": {}, "items": [], "dif": {}}

def run_scale(cols, tag):
    Xs = X[:, [item_cols.index(c) for c in cols]]
    b, sigma, n_it = irt.fit_rasch(Xs)
    theta, se = irt.wle(Xs, b)
    score = irt.to_scale(theta)
    infit, outfit = irt.fit_stats(Xs, theta, b)
    pb = irt.point_biserial(Xs, Xs.sum(axis=1))
    alpha = cronbach_alpha(pd.DataFrame(Xs))
    rel_wle = 1 - np.mean(se ** 2) / np.var(theta, ddof=1)
    return dict(b=b, sigma=sigma, theta=theta, se=se, score=score, infit=infit,
                outfit=outfit, pb=pb, alpha=alpha, rel=rel_wle, n_it=n_it, cols=cols)

# ----------------------------------- загальна шкала, галузеві та предметні лінії
runs = {"total": run_scale(item_cols, "Загальна")}
labels = {"total": "Загальна"}
for d, name in DOMAINS.items():
    runs[f"d{d}"] = run_scale([c for c in item_cols if c.startswith(d + ".")], name)
    labels[f"d{d}"] = name
for tag, name in SUBJECTS.items():
    runs[f"s_{tag}"] = run_scale([c for c in item_cols if subject_of(c) == tag], name)
    labels[f"s_{tag}"] = name

for tag, r in runs.items():
    students[f"score_{tag}"] = r["score"]
    students[f"theta_{tag}"] = r["theta"]
    students[f"se_{tag}"] = r["se"]
    label = labels[tag]
    report["scales"][tag] = {
        "label": label,
        "n_items": len(r["cols"]),
        "alpha": round(float(r["alpha"]), 3),
        "reliability_wle": round(float(r["rel"]), 3),
        "sigma_population": round(float(r["sigma"]), 3),
        "mean_pct": round(float(100 * X[:, [item_cols.index(c) for c in r["cols"]]].mean()), 2),
        "sd_theta": round(float(np.std(r["theta"], ddof=1)), 3),
        "mean_se": round(float(np.mean(r["se"])), 3),
        "em_iterations": int(r["n_it"]),
    }

# ------------------------------------------------------------ параметри завдань
tot = runs["total"]
key_map = dict(zip(key["item"], key["correct"]))
for j, c in enumerate(item_cols):
    d = c[0]
    report["items"].append({
        "item": c,
        "domain": DOMAINS[d],
        "subject": SUBJECTS[subject_of(c)],
        "task": re.sub(r"_\d\d$", "", c),
        "p_value": round(float(X[:, j].mean()), 3),
        "difficulty_b": round(float(tot["b"][j]), 3),
        "point_biserial": round(float(tot["pb"][j]), 3),
        "infit": round(float(tot["infit"][j]), 3),
        "outfit": round(float(tot["outfit"][j]), 3),
    })

# ------------------------------------------------------------------------ DIF
focal = (students["group"] == GROUP_ORDER[0]).values          # пілотні класи
ref_mask = (students["group"] == GROUP_ORDER[2]).values       # непілотні школи
sub = focal | ref_mask
mh = irt.mantel_haenszel(X[sub], focal[sub])
dif_rows = []
for c, (a, dl, cat) in zip(item_cols, mh):
    dif_rows.append({"item": c, "domain": DOMAINS[c[0]],
                     "alpha_mh": None if not np.isfinite(a) else round(float(a), 3),
                     "delta_ets": None if not np.isfinite(dl) else round(float(dl), 3),
                     "category": cat})
report["dif"] = {
    "focal": GROUP_ORDER[0], "reference": GROUP_ORDER[2],
    "counts": pd.Series([r["category"] for r in dif_rows]).value_counts().to_dict(),
    "flagged_C": [r for r in dif_rows if r["category"] == "C"],
    "all": dif_rows,
}

# перевірка стійкості: шкала без завдань категорії C
bad = [r["item"] for r in dif_rows if r["category"] == "C"]
if bad:
    clean_cols = [c for c in item_cols if c not in bad]
    r = run_scale(clean_cols, "clean")
    students["score_total_nodif"] = r["score"]
    report["dif"]["clean_scale"] = {
        "n_items": len(clean_cols),
        "corr_with_full": round(float(np.corrcoef(r["score"], tot["score"])[0, 1]), 4),
    }

# ------------------------------------- робастність: частковий кредит по завданнях
task_of = {c: re.sub(r"_\d\d$", "", c) for c in item_cols}
Xdf = pd.DataFrame(X, columns=item_cols)
task_df = pd.DataFrame({t: Xdf[[c for c in item_cols if task_of[c] == t]].sum(axis=1)
                        for t in dict.fromkeys(task_of.values())})
pc_total = task_df.div(task_df.max()).mean(axis=1)      # частка балу за завданнями
report["robustness_partial_credit"] = {
    "n_tasks": int(task_df.shape[1]),
    "corr_with_rasch": round(float(np.corrcoef(pc_total, tot["score"])[0, 1]), 4),
    "corr_pct_with_rasch": round(float(np.corrcoef(students["pct_total"], tot["score"])[0, 1]), 4),
}

# ------------------------------------------------------- рівні сформованості
q = np.quantile(students["score_total"], [0.25, 0.5, 0.75])
students["level"] = pd.cut(students["score_total"], [-np.inf, *q, np.inf],
                           labels=["Низький", "Середній нижній", "Середній верхній", "Високий"])

students.to_csv(OUT / "students.csv", index=False)
(OUT / "02_psychometrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

print(json.dumps(report["scales"], ensure_ascii=False, indent=2))
print("DIF:", report["dif"]["counts"])
print("robustness:", report["robustness_partial_credit"])
print(students.groupby("group")[["score_total", "pct_total"]].mean().round(1))
