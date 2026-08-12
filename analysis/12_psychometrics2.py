"""Етап 12. Перевірки, яких не робить α Кронбаха. -> output/12_psychometrics2.json

α показує лише узгодженість позицій і зростає з їхньою кількістю. Вона не доводить
ані одновимірності, ані незалежності позицій, ані точності для окремого учня.
Тут рахуємо саме це:
  · одновимірність     — PCA стандартизованих залишків Раша (критерій Лінакра)
  · локальна залежність — Q3 (кореляції залишків) у межах завдання та між завданнями
  · точність            — інформація тесту й похибка вимірювання на всіх рівнях
  · вплив проблемних позицій — шкала й ефект без них
"""
import json
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import OUT, cronbach_alpha
from irt import fit_rasch, wle

X = pd.read_csv(OUT / "items.csv")
st = pd.read_csv(OUT / "students.csv")
psy = json.loads((OUT / "02_psychometrics.json").read_text())
cols = [c for c in X.columns if c != "id"]
M = X[cols].values.astype(float)

b, sigma, _ = fit_rasch(M)
th, se = wle(M, b)
sd_th = float(th.std(ddof=1))
to_points = lambda x: x / sd_th * 100            # theta -> бали шкали 500/100

P = 1 / (1 + np.exp(-(th[:, None] - b[None, :])))
W = P * (1 - P)
Z = (M - P) / np.sqrt(W)                          # стандартизовані залишки

# ------------------------------------------------------ 1. одновимірність
Cres = np.nan_to_num(np.corrcoef(Z - Z.mean(0), rowvar=False), nan=0.0)
ev = np.sort(np.linalg.eigvalsh(Cres))[::-1]
dims = {
    "eigenvalues": [round(float(x), 2) for x in ev[:5]],
    "first_contrast": round(float(ev[0]), 2),
    "criterion": 2.0,
    "passes": bool(ev[0] < 2.0),
    "share_pct": round(float(ev[0] / len(cols) * 100), 2),
}

# кореляції між галузевими шкалами: скільки в тесті спільного
dcols = [f"theta_d{i}" for i in (1, 2, 3, 4)]
R = st[dcols].corr()
dims["domain_r"] = {a: {c: round(float(R.loc[a, c]), 3) for c in dcols} for a in dcols}
dims["domain_r_mean"] = round(float(R.values[np.triu_indices(4, 1)].mean()), 3)

# ------------------------------------------------- 2. локальна залежність (Q3)
task = np.array([re.sub(r"_\d\d$", "", c) for c in cols])
Rz = np.nan_to_num(np.corrcoef(Z, rowvar=False), nan=0.0)
iu = np.triu_indices(len(cols), 1)
same = task[iu[0]] == task[iu[1]]
q3 = Rz[iu]
q3s = q3 - q3.mean()                              # Q3* — центрована версія
local = {
    "n_pairs": int(len(q3)), "n_same_task": int(same.sum()),
    "mean_within_task": round(float(q3[same].mean()), 3),
    "mean_between_tasks": round(float(q3[~same].mean()), 3),
    "flagged_within": int((q3s[same] > 0.2).sum()),
    "flagged_between": int((q3s[~same] > 0.2).sum()),
    "n_multi_position_tasks": int(pd.Series(task).value_counts().gt(1).sum()),
}

# ------------------------------- 3. інформація тесту й похибка вимірювання
curve = []
for gtheta in np.arange(-3, 3.01, 0.5):
    p = 1 / (1 + np.exp(-(gtheta - b)))
    info = float((p * (1 - p)).sum())
    curve.append({"theta": round(float(gtheta), 1),
                  "score": round(500 + to_points(gtheta)),
                  "info": round(info, 1),
                  "se_points": round(to_points(1 / np.sqrt(info)), 1)})
precision = {
    "curve": curve,
    "mean_se_theta": round(float(se.mean()), 3),
    "mean_se_points": round(to_points(float(se.mean())), 1),
    "ci95_individual": round(1.96 * to_points(float(se.mean()))),
    "reliability_wle": psy["scales"]["total"]["reliability_wle"],
    # похибка середнього по групі падає як корінь із n
    "se_group_mean": {str(n): round(float(to_points(float(se.mean())) / np.sqrt(n)), 1)
                      for n in (20, 100, 1000, 2867)},
}

# --------------------------------- 4. α: довжина тесту й локальна залежність
alpha_pos = float(cronbach_alpha(X[cols]))
agg = pd.DataFrame({t: X[[c for c in cols if c == t or c.startswith(t + "_")]].sum(axis=1)
                    for t in pd.unique(task)})
alpha_task = float(cronbach_alpha(agg))


def spearman_brown(alpha, k):
    """α шкали, вкороченої (k < 1) або подовженої (k > 1) у k разів."""
    return k * alpha / (1 + (k - 1) * alpha)


alpha_len = {
    "alpha_positions": round(alpha_pos, 3), "n_positions": len(cols),
    "alpha_tasks": round(alpha_task, 3), "n_tasks": int(agg.shape[1]),
    "alpha_if_half_length": round(spearman_brown(alpha_pos, 0.5), 3),
    "alpha_if_20_items": round(spearman_brown(alpha_pos, 20 / len(cols)), 3),
}

# ---------------------------------------- 5. вплив проблемних позицій
it = pd.DataFrame(psy["items"])
bad = it[(it.point_biserial < 0.2)
         | (it.infit.round(2) < 0.8) | (it.infit.round(2) > 1.2)]["item"]
keep = [c for c in cols if c not in set(bad)]
Mk = X[keep].values.astype(float)
bk, _, _ = fit_rasch(Mk)
thk, sek = wle(Mk, bk)

d = st.copy()
d["score_short"] = 500 + (thk - thk.mean()) / thk.std(ddof=1) * 100
d["pilot"] = (d["group"] == "Пілотний клас").astype(float)
sub = d[d["group"] != "Непілотний клас у пілотній школі"].reset_index(drop=True)
eff = {}
for y in ("score_total", "score_short"):
    m = smf.ols(f"{y} ~ pilot + C(pair_id)", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["school_id"]})
    eff[y] = {"b": round(float(m.params["pilot"]), 1), "p": float(m.pvalues["pilot"])}

trim = {
    "n_removed": len(cols) - len(keep), "n_kept": len(keep),
    "n_low_pbis": int((it.point_biserial < 0.2).sum()),
    "n_negative_pbis": int((it.point_biserial <= 0).sum()),
    "n_misfit": int(((it.infit.round(2) < 0.8) | (it.infit.round(2) > 1.2)).sum()),
    "r_pearson": round(float(np.corrcoef(th, thk)[0, 1]), 4),
    "r_spearman": round(float(pd.Series(th).corr(pd.Series(thk), method="spearman")), 4),
    "alpha_kept": round(float(cronbach_alpha(X[keep])), 3),
    "effect_full": eff["score_total"], "effect_trimmed": eff["score_short"],
}

# ------------------------- 6. достатність сирого бала (проти хибного пояснення)
g = st.groupby("raw_total")["theta_total"]
mono = np.diff(g.first().sort_index().values)
sufficiency = {
    "max_theta_spread_within_raw": round(float((g.max() - g.min()).max()), 6),
    "monotone_steps": int((mono > 0).sum()), "n_steps": int(len(mono)),
    "r_pearson_raw": round(float(np.corrcoef(st.raw_total, st.theta_total)[0, 1]), 4),
    "r_pearson_pct": round(float(np.corrcoef(st.pct_total, st.score_total)[0, 1]), 4),
}

# скільки балів шкали додають 5 правильних відповідей у різних місцях шкали:
# посередині крок найменший, до обох країв росте (звідси «інтервальна, а не порядкова»)
gs = st.groupby("raw_total")["score_total"].first().sort_index()
def _score_at(r):
    return float(np.interp(r, gs.index.values, gs.values))
sufficiency["step5"] = {str(r): round(_score_at(r + 5) - _score_at(r))
                        for r in (15, 50, 85)}

out = {"dimensionality": dims, "local_dependence": local, "precision": precision,
       "alpha_context": alpha_len, "trimming": trim, "sufficiency": sufficiency}
(OUT / "12_psychometrics2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

print("=== ОДНОВИМІРНІСТЬ: 1-й контраст", dims["first_contrast"],
      "(критерій < 2,0) ->", "проходить" if dims["passes"] else "НЕ проходить")
print("   власні значення:", dims["eigenvalues"])
print("   середня кореляція між галузями:", dims["domain_r_mean"])
print("=== ЛОКАЛЬНА ЗАЛЕЖНІСТЬ: Q3 у межах завдання", local["mean_within_task"],
      "проти", local["mean_between_tasks"], "між завданнями")
print(f"   перевищують поріг: {local['flagged_within']} із {local['n_same_task']} "
      f"внутрішніх, {local['flagged_between']} із {local['n_pairs'] - local['n_same_task']} інших")
print("=== ТОЧНІСТЬ: середня похибка", precision["mean_se_points"],
      "бала; 95 % ДІ для одного учня ±", precision["ci95_individual"], "бала")
print("   похибка середнього по групі:", precision["se_group_mean"])
print("=== АЛЬФА: за позиціями", alpha_len["alpha_positions"],
      "| за завданнями", alpha_len["alpha_tasks"],
      "| була б на 20 позиціях", alpha_len["alpha_if_20_items"])
print("=== БЕЗ", trim["n_removed"], "ПРОБЛЕМНИХ ПОЗИЦІЙ: r =", trim["r_pearson"],
      "| ефект", trim["effect_full"]["b"], "->", trim["effect_trimmed"]["b"], "бала")
print("=== ДОСТАТНІСТЬ: розкид theta в межах сирого бала =",
      sufficiency["max_theta_spread_within_raw"],
      f"| монотонність {sufficiency['monotone_steps']}/{sufficiency['n_steps']}")
