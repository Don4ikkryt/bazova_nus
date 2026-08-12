"""Етап 5. Ефект НУШ: три сходинки доказовості. -> output/05_effects.json"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

from common import OUT, DOMAINS, GROUP_ORDER

df = pd.read_csv(OUT / "students.csv")
SCALES = {"score_total": "Загальна", **{f"score_d{d}": n for d, n in DOMAINS.items()}}
COVARS = ["female", "sen", "idp", "abroad", "remote", "online_test",
          "books", "HOME", "parent_edu", "UKRLANG"]


def prep(d, covars=COVARS):
    """Медіанна імпутація + індикатор пропуску (частка пропусків фіксується у звіті)."""
    d = d.copy()
    used, miss_share = [], {}
    for c in covars:
        share = float(d[c].isna().mean())
        miss_share[c] = round(share, 3)
        if share > 0:
            d[c + "_m"] = d[c].isna().astype(float)
            if share > 0.02:
                used.append(c + "_m")
        d[c] = d[c].fillna(d[c].median())
        used.append(c)
    return d, used, miss_share


def fit(formula, data, cluster="school_id"):
    m = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster]})
    return m


def extract(m, term, sd):
    b = m.params[term]
    se = m.bse[term]
    return {
        "b": round(float(b), 2),
        "se": round(float(se), 2),
        "ci": [round(float(b - 1.96 * se), 2), round(float(b + 1.96 * se), 2)],
        "sigma": round(float(b / sd), 3),
        "sigma_ci": [round(float((b - 1.96 * se) / sd), 3),
                     round(float((b + 1.96 * se) / sd), 3)],
        "t": round(float(m.tvalues[term]), 2),
        "p": float(m.pvalues[term]),
        "n": int(m.nobs),
        "r2": round(float(m.rsquared), 3),
    }


def bh(pvals):
    """Поправка Бенджаміні–Хохберга."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        val = min(prev, p[idx] * m / (i + 1))
        ranked[idx] = val
        prev = val
    return ranked


out = {"ladders": {}, "by_domain": {}, "icc": {}, "extra": {}}

for col, label in SCALES.items():
    sd = df[col].std(ddof=1)
    res = {}

    # --- сходинка 1: гола різниця, пілотний клас vs непілотна школа
    d1 = df[df["group"].isin([GROUP_ORDER[0], GROUP_ORDER[2]])].copy()
    d1["pilot"] = (d1["group"] == GROUP_ORDER[0]).astype(float)
    res["1_raw"] = extract(fit(f"{col} ~ pilot", d1), "pilot", sd)

    # --- сходинка 2: у межах 50 пар шкіл + індивідуальні коваріати
    d2, used, miss = prep(d1)
    res["2_pair_fe"] = extract(
        fit(f"{col} ~ pilot + C(pair_id) + " + " + ".join(used), d2), "pilot", sd)
    res["2_pair_fe_nocov"] = extract(
        fit(f"{col} ~ pilot + C(pair_id)", d2), "pilot", sd)

    # --- сходинка 3: усередині однієї пілотної школи
    d3 = df[df["pilot_school"] == 1].copy()
    both = d3.groupby("school_id")["class_pilot"].nunique().eq(2)
    d3 = d3[d3["school_id"].isin(both[both].index)].copy()
    d3["pilot"] = (d3["class_pilot"] == "Пілотний").astype(float)
    d3p, used3, _ = prep(d3)
    res["3_school_fe"] = extract(
        fit(f"{col} ~ pilot + C(school_id) + " + " + ".join(used3), d3p), "pilot", sd)
    res["3_school_fe_nocov"] = extract(
        fit(f"{col} ~ pilot + C(school_id)", d3p), "pilot", sd)
    res["3_n_schools"] = int(d3["school_id"].nunique())

    # --- додатково: ефект «пілотна школа» (будь-який клас) у межах пари
    d4 = df.copy()
    d4["pilot_sch"] = d4["pilot_school"].astype(float)
    d4p, used4, _ = prep(d4)
    res["school_level"] = extract(
        fit(f"{col} ~ pilot_sch + C(pair_id) + " + " + ".join(used4), d4p), "pilot_sch", sd)

    res["missing_share"] = miss
    out["ladders" if col == "score_total" else "by_domain"] = \
        res if col == "score_total" else {**out.get("by_domain", {}), label: res}

# поправка на множинні порівняння по галузях
for step in ["1_raw", "2_pair_fe", "3_school_fe"]:
    ps = [out["by_domain"][d][step]["p"] for d in out["by_domain"]]
    for d, q in zip(out["by_domain"], bh(ps)):
        out["by_domain"][d][step]["p_bh"] = float(q)

# ------------------------------------------------------- ICC (школа / клас)
mdf = df.dropna(subset=["score_total"]).copy()


def icc(group, fe=None, data=None):
    d = mdf if data is None else data
    m = MixedLM.from_formula(f"score_total ~ {fe or '1'}", groups=group, data=d).fit(reml=True)
    vb, vw = float(m.cov_re.iloc[0, 0]), float(m.scale)
    return vb / (vb + vw), vb, vw


r_school, var_school, var_resid = icc("school_id")
out["icc"]["school"] = round(r_school, 3)
out["icc"]["class"] = round(icc("class_id")[0], 3)

# Скільки з міжшкільної різниці — це артефакт умов складання. Онлайн-складання
# майже повністю згруповане по школах, тож воно прямо роздуває ICC.
out["icc"]["school_adj_mode"] = round(icc("school_id", fe="online_test")[0], 3)
out["icc"]["school_adj_mode_region"] = round(
    icc("school_id", fe="online_test + C(region) + rural")[0], 3)
out["icc"]["school_offline_only"] = round(
    icc("school_id", data=mdf[mdf["online_test"] == 0])[0], 3)

sch_online = mdf.groupby("school_id")["online_test"].mean()
sch_mean = mdf.groupby("school_id")["score_total"].mean()
out["icc"]["mode_clustering"] = {
    "schools_all_offline": int((sch_online == 0).sum()),
    "schools_all_online": int((sch_online == 1).sum()),
    "schools_mixed": int(((sch_online > 0) & (sch_online < 1)).sum()),
    "r_online_share_school_mean": round(float(np.corrcoef(sch_online, sch_mean)[0, 1]), 3),
}
# наочний переклад ICC: середня різниця балів двох навмання взятих учнів
gap = lambda sd: round(float(2 * sd / np.sqrt(np.pi)))
out["icc"]["typical_gap"] = {
    "two_random_students": gap(mdf["score_total"].std(ddof=1)),
    "two_from_same_school": gap(np.sqrt(var_resid)),
}
out["icc"]["n_classes_per_school"] = {
    str(k): int(v) for k, v in
    mdf.groupby("school_id")["class_id"].nunique().value_counts().sort_index().items()}

# ------------------------------------------------------ порівняння підгруп 1 vs 2
d12 = df[df["pilot_school"] == 1].copy()
d12["pilot"] = (d12["class_pilot"] == "Пілотний").astype(float)
out["extra"]["pilot_vs_nonpilot_class_all_pilot_schools"] = extract(
    fit("score_total ~ pilot", d12), "pilot", df["score_total"].std(ddof=1))

# розкид ефекту по парах (для «лісової» картинки)
pair_rows = []
for pid, sub in df[df["group"].isin([GROUP_ORDER[0], GROUP_ORDER[2]])].groupby("pair_id"):
    a = sub[sub["group"] == GROUP_ORDER[0]]["score_total"]
    b = sub[sub["group"] == GROUP_ORDER[2]]["score_total"]
    if len(a) < 3 or len(b) < 3:
        continue
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    pair_rows.append({"pair": pid, "diff": round(float(a.mean() - b.mean()), 1),
                      "se": round(float(se), 1), "n_pilot": int(len(a)), "n_ctrl": int(len(b)),
                      "region": sub["region"].iloc[0],
                      "urban": sub["urban"].iloc[0]})
out["pairs"] = sorted(pair_rows, key=lambda r: r["diff"])
out["pairs_summary"] = {
    "n_pairs": len(pair_rows),
    "share_positive": round(float(np.mean([r["diff"] > 0 for r in pair_rows])), 3),
    "median_diff": round(float(np.median([r["diff"] for r in pair_rows])), 1),
}

# гетерогенність: чи однаковий ефект у різних підгрупах
het = {}
for name, series in {"Місцевість": df["urban"],
                     "Стать": df["female"].map({0: "Хлопці", 1: "Дівчата"}),
                     "Книг удома": pd.cut(df["books"], [0, 3, 6], labels=["до 25", "понад 25"]),
                     "Формат": df["remote"].map({0: "Очно", 0.5: "Змішано", 1: "Дистанційно"})}.items():
    tmp = df.assign(_c=series)
    lev = {}
    for k, sub in tmp.groupby("_c", observed=True):
        sub = sub[sub["group"].isin([GROUP_ORDER[0], GROUP_ORDER[2]])].copy()
        if sub["school_id"].nunique() < 10 or len(sub) < 100:
            continue
        sub["pilot"] = (sub["group"] == GROUP_ORDER[0]).astype(float)
        lev[str(k)] = extract(fit("score_total ~ pilot", sub), "pilot",
                              df["score_total"].std(ddof=1))
    het[name] = lev
out["heterogeneity"] = het

# --------------------------------------- умови складання: очно проти онлайн
sd = df["score_total"].std(ddof=1)
mode = {"share_online": round(float(df["online_test"].mean()), 3),
        "share_online_by_group": {g: round(float(s["online_test"].mean()), 3)
                                  for g, s in df.groupby("group")},
        "mean_online": round(float(df[df.online_test == 1]["score_total"].mean()), 1),
        "mean_offline": round(float(df[df.online_test == 0]["score_total"].mean()), 1)}
mode["raw"] = extract(fit("score_total ~ online_test", df), "online_test", sd)
# найчистіша перевірка: ті самі однокласники, різний спосіб складання
mixed = df.groupby("class_id")["online_test"].nunique().eq(2)
dm2 = df[df["class_id"].isin(mixed[mixed].index)]
mode["within_class"] = extract(
    smf.ols("score_total ~ online_test + C(class_id)", data=dm2).fit(
        cov_type="cluster", cov_kwds={"groups": dm2["school_id"]}), "online_test", sd)
mode["within_class_n_classes"] = int(dm2["class_id"].nunique())
out["test_mode"] = mode

# --------------------------------- стійкість: лише ті, хто складав очно в школі
off = df[df["online_test"] == 0].copy()
rob = {}
d1 = off[off["group"].isin([GROUP_ORDER[0], GROUP_ORDER[2]])].copy()
d1["pilot"] = (d1["group"] == GROUP_ORDER[0]).astype(float)
d1p, used, _ = prep(d1)
rob["1_raw"] = extract(fit("score_total ~ pilot", d1p), "pilot", sd)
rob["2_pair_fe"] = extract(
    fit("score_total ~ pilot + C(pair_id) + " + " + ".join(used), d1p), "pilot", sd)
d3 = off[off["pilot_school"] == 1].copy()
both = d3.groupby("school_id")["class_pilot"].nunique().eq(2)
d3 = d3[d3["school_id"].isin(both[both].index)].copy()
d3["pilot"] = (d3["class_pilot"] == "Пілотний").astype(float)
d3p, used3, _ = prep(d3)
rob["3_school_fe"] = extract(
    fit("score_total ~ pilot + C(school_id) + " + " + ".join(used3), d3p), "pilot", sd)
rob["n"] = int(len(off))
out["robustness_offline_only"] = rob

# ------------------------- той самий ефект, але в кількості правильних відповідей
raw = {}
for col, unit in [("raw_total", "правильних позицій зі 103"), ("pct_total", "% правильних")]:
    sd_c = df[col].std(ddof=1)
    d1 = df[df["group"].isin([GROUP_ORDER[0], GROUP_ORDER[2]])].copy()
    d1["pilot"] = (d1["group"] == GROUP_ORDER[0]).astype(float)
    d1p, used, _ = prep(d1)
    d3 = df[df["pilot_school"] == 1].copy()
    both = d3.groupby("school_id")["class_pilot"].nunique().eq(2)
    d3 = d3[d3["school_id"].isin(both[both].index)].copy()
    d3["pilot"] = (d3["class_pilot"] == "Пілотний").astype(float)
    d3p, used3, _ = prep(d3)
    raw[col] = {
        "unit": unit,
        "mean_by_group": {g: round(float(s[col].mean()), 2) for g, s in df.groupby("group")},
        "1_raw": extract(fit(f"{col} ~ pilot", d1p), "pilot", sd_c),
        "2_pair_fe": extract(
            fit(f"{col} ~ pilot + C(pair_id) + " + " + ".join(used), d1p), "pilot", sd_c),
        "3_school_fe": extract(
            fit(f"{col} ~ pilot + C(school_id) + " + " + ".join(used3), d3p), "pilot", sd_c),
    }
out["in_correct_answers"] = raw

# ------------------------------------------------- амплітуда: наскільки це багато
sd_all = df["score_total"].std(ddof=1)
sch = df.groupby("school_id")["score_total"].mean()
cls = df.groupby("class_id")["score_total"].mean()
diffs = np.array([r["diff"] for r in pair_rows])
cuts = json.loads((OUT / "04_descriptive.json").read_text())["cuts"]
pe = cuts["Освіта батьків"]["levels"]

amp = {
    "effect_points": out["ladders"]["2_pair_fe"]["b"],
    "effect_sigma": out["ladders"]["2_pair_fe"]["sigma"],
    # розкид самого ефекту між парами
    "pair_diff_sd": round(float(diffs.std(ddof=1)), 1),
    "pair_diff_iqr": [round(float(np.percentile(diffs, 25)), 1),
                      round(float(np.percentile(diffs, 75)), 1)],
    "pair_diff_range": [round(float(diffs.min()), 1), round(float(diffs.max()), 1)],
    "share_pairs_negative": round(float((diffs < 0).mean()), 3),
    "share_pairs_beyond_20": round(float((np.abs(diffs) > 20).mean()), 3),
    # інтервал, у якому лежить ефект для випадково взятої пари
    "prediction_interval": [round(float(diffs.mean() - 1.96 * diffs.std(ddof=1)), 1),
                            round(float(diffs.mean() + 1.96 * diffs.std(ddof=1)), 1)],
    # з чим порівняти
    "benchmarks": {
        "Ефект пілоту НУШ (між школами)": out["ladders"]["2_pair_fe"]["sigma"],
        "Ефект пілоту НУШ (усередині школи)": out["ladders"]["3_school_fe"]["sigma"],
        "Розрив за освітою батьків (базова середня → вища)":
            round(float((pe["Вища"]["mean"] - pe["Базова середня"]["mean"]) / sd_all), 3),
        "Розрив між найслабшою і найсильнішою школою":
            round(float((sch.max() - sch.min()) / sd_all), 3),
        "Розрив між 10 % найслабших і 10 % найсильніших шкіл":
            round(float((sch.quantile(.9) - sch.quantile(.1)) / sd_all), 3),
        "Різниця між очним і онлайновим складанням тесту":
            round(float(out["test_mode"]["raw"]["sigma"]), 3),
    },
    "school_means": {"min": round(float(sch.min()), 1), "max": round(float(sch.max()), 1),
                     "sd": round(float(sch.std(ddof=1)), 1),
                     "p10": round(float(sch.quantile(.1)), 1),
                     "p90": round(float(sch.quantile(.9)), 1)},
    "class_means": {"min": round(float(cls.min()), 1), "max": round(float(cls.max()), 1),
                    "sd": round(float(cls.std(ddof=1)), 1)},
    # скільки перекриття між групами: частка непілотних учнів вище медіани пілотних
    "overlap": {
        "share_control_above_pilot_median": round(float(
            (df[df.group == GROUP_ORDER[2]]["score_total"] >
             df[df.group == GROUP_ORDER[0]]["score_total"].median()).mean()), 3),
        "share_pilot_above_control_median": round(float(
            (df[df.group == GROUP_ORDER[0]]["score_total"] >
             df[df.group == GROUP_ORDER[2]]["score_total"].median()).mean()), 3),
    },
}
# ймовірність, що випадковий учень пілотного класу випередить випадкового з контролю
a = df[df.group == GROUP_ORDER[0]]["score_total"].values
b = df[df.group == GROUP_ORDER[2]]["score_total"].values
amp["probability_of_superiority"] = round(float(
    (a[:, None] > b[None, :]).mean()), 3)
out["amplitude"] = amp

(OUT / "05_effects.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out["ladders"], ensure_ascii=False, indent=1))
print("ICC:", out["icc"])
print("пари:", out["pairs_summary"])
