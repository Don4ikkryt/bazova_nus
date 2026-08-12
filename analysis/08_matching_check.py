"""Додаткова перевірка: чи не є відповіді на завдання з кількох позицій
типовою «незміненою» послідовністю А, Б, В, Г… (тобто фактично неспробою).
-> output/08_matching.json
"""
import json
import re

import pandas as pd

from common import OUT, XLSX, norm_label

test = pd.read_excel(XLSX, sheet_name="Дані тестування")
key = pd.read_csv(OUT / "key.csv")
part = test[test["Тетсування пройдено (Yes/No)"] == "Yes"]

ALPHABET = list("АБВГДЕЄЖЗИІ")
tasks = {}
for item in key["item"]:
    tasks.setdefault(re.sub(r"_\d\d$", "", item), []).append(item)

rows = []
for task, cols in tasks.items():
    if len(cols) < 2:
        continue
    cols = sorted(cols)
    given = part[cols].apply(lambda s: s.map(norm_label))
    identity = [a.lower() for a in ALPHABET[:len(cols)]]
    is_identity = (given.values == identity).all(axis=1)
    correct = [norm_label(key[key["item"] == c]["correct"].iloc[0]) for c in cols]
    n_correct = (given.values == correct).sum(axis=1)
    rows.append({
        "task": task,
        "domain": task[0],
        "n_positions": len(cols),
        "share_identity_sequence": round(float(is_identity.mean()), 3),
        "key_is_identity": correct == identity,
        "mean_share_correct": round(float((n_correct / len(cols)).mean()), 3),
        "share_all_correct": round(float((n_correct == len(cols)).mean()), 3),
    })

rows.sort(key=lambda r: -r["share_identity_sequence"])
flagged = [r for r in rows if r["share_identity_sequence"] > 0.15]
out = {"tasks": rows, "n_multi_tasks": len(rows), "flagged": flagged}

# --- стійкість: шкала без сумнівних завдань, ті самі три сходинки
import numpy as np
import statsmodels.formula.api as smf
import irt
from common import GROUP_ORDER

drop = [c for r in flagged for c in tasks[r["task"]]]
items = pd.read_csv(OUT / "items.csv")
students = pd.read_csv(OUT / "students.csv")
keep = [c for c in items.columns if c != "id" and c not in drop]
Xk = items[keep].values.astype(float)
b, _, _ = irt.fit_rasch(Xk)
theta, _ = irt.wle(Xk, b)
students["score_clean"] = irt.to_scale(theta)
sd = students["score_clean"].std(ddof=1)

d1 = students[students["group"].isin([GROUP_ORDER[0], GROUP_ORDER[2]])].copy()
d1["pilot"] = (d1["group"] == GROUP_ORDER[0]).astype(float)
d3 = students[students["pilot_school"] == 1].copy()
both = d3.groupby("school_id")["class_pilot"].nunique().eq(2)
d3 = d3[d3["school_id"].isin(both[both].index)].copy()
d3["pilot"] = (d3["class_pilot"] == "Пілотний").astype(float)


def est(formula, data, term="pilot"):
    m = smf.ols(formula, data=data).fit(cov_type="cluster",
                                        cov_kwds={"groups": data["school_id"]})
    return {"sigma": round(float(m.params[term] / sd), 3), "p": float(m.pvalues[term])}


out["robustness_without_flagged"] = {
    "n_positions_dropped": len(drop),
    "corr_with_full": round(float(np.corrcoef(students["score_clean"],
                                              students["score_total"])[0, 1]), 4),
    "1_raw": est("score_clean ~ pilot", d1),
    "2_pair_fe": est("score_clean ~ pilot + C(pair_id)", d1),
    "3_school_fe": est("score_clean ~ pilot + C(school_id)", d3),
}
(OUT / "08_matching.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out["robustness_without_flagged"], ensure_ascii=False))
for r in rows:
    print(f"{r['task']:22s} поз={r['n_positions']}  «А,Б,В,Г…»={r['share_identity_sequence']:.1%}"
          f"  ключ=послідовність: {r['key_is_identity']}  сер.частка правильних={r['mean_share_correct']:.2f}")
