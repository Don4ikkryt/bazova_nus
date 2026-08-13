"""Етап 4. Описова статистика та розрізи. -> output/04_descriptive.json"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, DOMAINS, GROUP_ORDER, SUBJECTS

df = pd.read_csv(OUT / "students.csv")
SCALES = {"score_total": "Загальна",
          **{f"score_d{d}": n for d, n in DOMAINS.items()},
          **{f"score_s_{t}": n for t, n in SUBJECTS.items()}}


def summarize(g, col="score_total"):
    v = g[col].dropna()
    if len(v) < 2:
        return None
    se = v.std(ddof=1) / np.sqrt(len(v))
    return {
        "n": int(len(v)),
        "mean": round(float(v.mean()), 1),
        "sd": round(float(v.std(ddof=1)), 1),
        "se": round(float(se), 1),
        "ci": [round(float(v.mean() - 1.96 * se), 1), round(float(v.mean() + 1.96 * se), 1)],
        "p25": round(float(v.quantile(0.25)), 1),
        "median": round(float(v.median()), 1),
        "p75": round(float(v.quantile(0.75)), 1),
        "pct": round(float(g.loc[v.index, col.replace("score", "pct")].mean()), 1)
        if col.replace("score", "pct") in g else None,
    }


out = {"overall": {}, "by_group": {}, "cuts": {}, "distribution": {}, "grades": {}}

for col, label in SCALES.items():
    out["overall"][col] = {"label": label, **summarize(df, col)}
    out["by_group"][col] = {"label": label,
                            **{g: summarize(sub, col) for g, sub in df.groupby("group")}}

# ------------------------------------------------------------------ розрізи
CUTS = {
    "Стать": df["female"].map({0: "Хлопці", 1: "Дівчата"}),
    "Місцевість": df["urban"],
    "Тип закладу": df["school_type"],
    "Формат навчання": df["remote"].map({0: "Очно", 0.5: "Змішано", 1: "Дистанційно"}),
    "Статус ВПО": df["idp"].map({0: "Не ВПО", 1: "ВПО"}),
    "Особливі освітні потреби": df["sen"].map({0: "Без ООП", 1: "З ООП"}),
    "Перебування": df["abroad"].map({0: "В Україні", 1: "За кордоном"}),
    "Спосіб тестування": df["online_test"].map({0: "Очно", 1: "Онлайн"}),
    "Книг удома": pd.cut(df["books"], [0, 2, 3, 4, 6],
                         labels=["до 10", "11–25", "26–100", "понад 100"]),
    "Освіта батьків": df["parent_edu"].map({1: "Базова середня", 2: "Повна середня",
                                            3: "Профтех / фахова", 4: "Вища",
                                            5: "Науковий ступінь"}),
    "Регіон": df["region"],
}
for name, series in CUTS.items():
    tmp = df.assign(_c=series)
    rows = {str(k): summarize(sub) for k, sub in tmp.groupby("_c", observed=True)}
    rows = {k: v for k, v in rows.items() if v}
    # той самий розріз усередині груп дизайну
    inter = {}
    for k, sub in tmp.groupby("_c", observed=True):
        inter[str(k)] = {g: (summarize(s2) or {}).get("mean")
                         for g, s2 in sub.groupby("group")}
    out["cuts"][name] = {"levels": rows, "by_group": inter}

# ------------------------------------------------------------- розподіл балів
hist, edges = np.histogram(df["score_total"], bins=np.arange(200, 820, 20))
out["distribution"]["overall"] = {"edges": edges.tolist(), "counts": hist.tolist()}
for g, sub in df.groupby("group"):
    h, _ = np.histogram(sub["score_total"], bins=edges)
    out["distribution"][g] = (h / h.sum()).round(4).tolist()
out["distribution"]["levels"] = {
    g: sub["level"].value_counts(normalize=True).round(3).to_dict()
    for g, sub in df.groupby("group")}

# --------------------------------------------------- оцінки проти результату
gr = {"Мовно-літературна": ("GRADE_L", "score_d1"), "Математична": ("GRADE_M", "score_d2"),
      "Природнича": ("GRADE_S", "score_d3"), "Громадянсько-історична": ("GRADE_H", "score_d4"),
      "Загалом": ("GRADE_ALL", "score_total")}
for label, (g, s) in gr.items():
    d = df[[g, s, "group"]].dropna()
    out["grades"][label] = {
        "r_all": round(float(np.corrcoef(d[g], d[s])[0, 1]), 3),
        "n": int(len(d)),
        "mean_grade": round(float(d[g].mean()), 2),
        "by_group": {gg: round(float(np.corrcoef(sub[g], sub[s])[0, 1]), 3)
                     for gg, sub in d.groupby("group") if len(sub) > 30},
    }
# середній бал тесту в межах однакової річної оцінки
d = df[["GRADE_ALL", "score_total", "group"]].dropna()
d["gbin"] = pd.cut(d["GRADE_ALL"], [0, 6, 8, 10, 12], labels=["до 6", "6–8", "8–10", "10–12"])
out["grades"]["same_grade_different_score"] = {
    str(k): {gg: round(float(s2["score_total"].mean()), 1) for gg, s2 in sub.groupby("group")}
    | {"n": int(len(sub))}
    for k, sub in d.groupby("gbin", observed=True)}

# ----------------------------------------------------- зусилля та мотивація
out["effort"] = {
    "mean_effort": {g: round(float(sub["effort"].mean()), 2) for g, sub in df.groupby("group")},
    "mean_effort_if_graded": {g: round(float(sub["effort_if_graded"].mean()), 2)
                              for g, sub in df.groupby("group")},
    "corr_effort_score": round(float(df[["effort", "score_total"]].dropna().corr().iloc[0, 1]), 3),
    "gap": round(float((df["effort_if_graded"] - df["effort"]).mean()), 2),
}

out["participation"] = {
    "n_participants": int(len(df)),
    "by_group": df["group"].value_counts().to_dict(),
    "n_schools": int(df["school_id"].nunique()),
    "n_classes": int(df["class_id"].nunique()),
    "schools_with_both_class_types": int(
        df[df.pilot_school == 1].groupby("school_id")["class_pilot"].nunique().eq(2).sum()),
}

(OUT / "04_descriptive.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out["by_group"]["score_total"], ensure_ascii=False, indent=1))
print("грамотність оцінок:", json.dumps(out["grades"], ensure_ascii=False)[:600])
print("участь:", out["participation"])
