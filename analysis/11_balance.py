"""Етап 11. Баланс підібраних пар шкіл. -> output/11_balance.json

Пари формували автори дослідження — ідентифікатор пари прийшов у робочій книзі,
алгоритм підбору нам невідомий. Тому тут два завдання:
  1) реконструювати, за чим пари фактично збігаються;
  2) виміряти залишковий дисбаланс за характеристиками, відомими до пілоту,
     і перевірити, чи він пояснює різницю в балах 2026 року.

Основна метрика — стандартизована різниця середніх (SMD). Звичайні p-value для
перевірки балансу не використовуємо: вони залежать від розміру вибірки, а не від
величини дисбалансу.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import OUT

st = pd.read_csv(OUT / "students.csv")
sb = pd.read_csv(OUT / "school_baseline.csv")

# ---------------------------------------------- 1. характеристики на рівні школи
comp = st.groupby("school_id").agg(
    school_type=("school_type", "first"),
    n_tested=("id", "size"), n_classes=("class_id", "nunique"),
    female=("female", "mean"), sen=("sen", "mean"), idp=("idp", "mean"),
    abroad=("abroad", "mean"), online=("online_test", "mean"),
    remote=("remote", "mean"), parent_edu=("parent_edu", "mean"),
    HOME=("HOME", "mean"), books=("books", "mean"), ladder=("ladder", "mean"),
    UKRLANG=("UKRLANG", "mean"), WAR=("WAR", "mean"))
sb = sb.merge(comp, on="school_id")
sb["gymnasium"] = (sb["school_type"] == "гімназія").astype(float)
sb["rural"] = (sb["urban"] == "Сільська").astype(float)
sb["size_2018"] = sb["n_2018"]
sb["size_2021"] = sb["n_2021"]
# кількість протестованих у закладі порівнювати не можна: у пілотних школах за
# дизайном тестували два класи, у непілотних — один. Порівнюємо розмір класу.
sb["class_size"] = sb["n_tested"] / sb["n_classes"]

# --------------------------------------- 2. за чим пари фактично збігаються
g = sb.groupby("pair_id")
matched_on = {
    "Область": {"same": int(g["region"].nunique().eq(1).sum()), "of": g.ngroups},
    "Тип місцевості (місто / село)": {"same": int(g["urban"].nunique().eq(1).sum()),
                                      "of": g.ngroups},
    "Тип закладу (гімназія / ліцей)": {"same": int(g["gymnasium"].nunique().eq(1).sum()),
                                       "of": g.ngroups},
}

# ------------------------------------------------------------- 3. таблиця балансу
# «Час» — коли ознака визначилася відносно старту пілоту (2018 р., 5 клас).
# «Блок» — за яким аргументом ознаку читати; у звіті рядки згруповані саме так,
# бо сортування за |SMD| ставить поруч речі різної природи.
VARS = [
    ("rural", "Сільська місцевість", "до", "частка", "match"),
    ("gymnasium", "Гімназія (проти ліцею)", "до", "частка", "match"),
    ("parent_edu", "Освіта батьків", "не залежить", "1–5", "home"),
    ("HOME", "Домашні ресурси", "не залежить", "0–1", "home"),
    ("books", "Книг удома", "не залежить", "1–5", "home"),
    ("ladder", "Самооцінка достатку родини", "не залежить", "1–10", "home"),
    ("UKRLANG", "Українська у щоденному спілкуванні", "не залежить", "1–4", "home"),
    ("sen", "Частка учнів з ООП", "не залежить", "частка", "composition"),
    ("idp", "Частка ВПО", "після 2022", "частка", "composition"),
    ("abroad", "Частка за кордоном", "після 2022", "частка", "composition"),
    ("female", "Частка дівчат", "не залежить", "частка", "composition"),
    ("WAR", "Обставин війни, що заважали", "після 2022", "0–6", "conditions"),
    ("class_size", "Учнів у протестованому класі", "не залежить", "осіб", "conditions"),
    ("remote", "Дистанційний формат", "після 2022", "0–1", "conditions"),
    ("online", "Частка онлайн-складання", "умова вимірювання", "частка", "conditions"),
]

BLOCKS = {
    "match": "За чим пари підібрані (відоме до пілоту)",
    "home": "Соціально-економічне тло родин",
    "composition": "Склад учнів",
    "conditions": "Умови навчання й вимірювання 2026",
}

# Історія ЗНО/НМТ у таблицю балансу не входить: ці ознаки відомі не для всіх шкіл,
# причому для пілотних і непілотних — для різних. SMD там порівнює дві різні неповні
# вибірки, а не підібрані групи. Рахуємо їх окремо, щоб навести в застереженні.
VARS_EXCLUDED = [
    ("size_2018", "Випускників у базі ЗНО-2018", "до", "осіб", "zno"),
    ("z_2018", "Результат ЗНО 2018", "до", "σ", "zno"),
    ("z_2021", "Результат ЗНО 2021", "до", "σ", "zno"),
    ("baseline_pre", "Індекс до пілоту (2018 + 2021)", "до", "σ", "zno"),
    ("baseline_all", "Індекс за 4 роки (2018–2025)", "частково", "σ", "zno"),
]


def smd_row(v, label, when, unit, block):
    a = sb.loc[sb.pilot_school == 1, v].dropna()
    b = sb.loc[sb.pilot_school == 0, v].dropna()
    # об'єднане SD — знаменник SMD; рахуємо на школах, а не на учнях
    sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    smd = float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0
    piv = (sb[["pair_id", "pilot_school", v]].dropna()
           .pivot_table(index="pair_id", columns="pilot_school", values=v).dropna())
    dif = (piv[1] - piv[0]) if len(piv) else pd.Series(dtype=float)
    return {
        "var": v, "label": label, "when": when, "unit": unit, "block": block,
        "n_pilot": int(len(a)), "n_nonpilot": int(len(b)),
        "mean_pilot": round(float(a.mean()), 3), "mean_nonpilot": round(float(b.mean()), 3),
        "sd_pooled": round(sd, 3),
        "smd": round(smd, 3),
        "n_pairs": int(len(piv)),
        # розкид внутрішньопарних різниць — наскільки тісно підібрані пари
        "pair_spread_sd": round(float(dif.std(ddof=1) / sd), 2) if len(piv) > 1 and sd > 0 else None,
        "pair_share_small": round(float((dif.abs() / sd < 0.25).mean()), 3) if len(piv) else None,
    }


rows = [smd_row(*a) for a in VARS]
rows_excluded = [smd_row(*a) for a in VARS_EXCLUDED]

n_arm = int(sb.pilot_school.sum())
se_smd = float(np.sqrt(2 / n_arm))
summary = {
    "n_schools_per_arm": n_arm,
    "se_smd": round(se_smd, 3),
    "ci95_smd": round(1.96 * se_smd, 2),
    "n_vars": len(rows),
    "n_under_0.1": int(sum(abs(r["smd"]) < 0.1 for r in rows)),
    "n_over_0.25": int(sum(abs(r["smd"]) > 0.25 for r in rows)),
    "n_over_ci": int(sum(abs(r["smd"]) > 1.96 * se_smd for r in rows)),
    # чи систематично зміщені в один бік ознаки, що описують силу контингенту
    "background_vars": ["parent_edu", "HOME", "books", "ladder"],
}
summary["background_favor_pilot"] = int(sum(
    r["smd"] > 0 for r in rows if r["var"] in summary["background_vars"]))
summary["background_n"] = len(summary["background_vars"])

# ------------- 4. чи пояснює дисбаланс різницю 2026 року: внутрішньопарні зв'язки
piv = sb.pivot_table(index="pair_id", columns="pilot_school",
                     values=["score_2026", "baseline_pre", "baseline_all",
                             "size_2018", "gymnasium"])
link = {}
for v, lab in [("baseline_pre", "Індекс до пілоту"), ("baseline_all", "Індекс за 4 роки"),
               ("size_2018", "Розмір закладу"), ("gymnasium", "Тип закладу")]:
    q = pd.DataFrame({"h": piv[v][1] - piv[v][0],
                      "s": piv["score_2026"][1] - piv["score_2026"][0]}).dropna()
    if len(q) > 4:
        link[v] = {"label": lab, "n_pairs": int(len(q)),
                   "r": round(float(q.corr().iloc[0, 1]), 3)}

# ------------------------------- 5. чи змінює врахування дисбалансу оцінку ефекту
d = st.merge(sb[["school_id", "baseline_all", "size_2018", "gymnasium"]],
             on="school_id", how="left")
d["pilot"] = (d["group"] == "Пілотний клас").astype(float)
d = d[d["group"] != "Непілотний клас у пілотній школі"].reset_index(drop=True)
BASE = ["female", "sen", "idp", "abroad", "online_test", "books", "HOME",
        "parent_edu", "ladder", "UKRLANG", "WAR"]
extra_terms = {}
for c in BASE + ["baseline_all", "size_2018", "gymnasium"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
    if d[c].isna().any():
        d[c + "_m"] = d[c].isna().astype(float)
        d[c] = d[c].fillna(d[c].median())
        extra_terms[c] = f"{c} + {c}_m"
    else:
        extra_terms[c] = c
cov = " + ".join(extra_terms[c] for c in BASE)

SPECS = [
    ("Як у звіті (сходинка 2)", ""),
    ("+ розмір закладу", " + " + extra_terms["size_2018"]),
    ("+ історія ЗНО / НМТ", " + " + extra_terms["baseline_all"]),
    ("+ тип закладу", " + " + extra_terms["gymnasium"]),
    ("+ усі три разом", " + " + " + ".join(
        extra_terms[c] for c in ("size_2018", "baseline_all", "gymnasium"))),
]
sensitivity = []
for lab, extra in SPECS:
    m = smf.ols(f"score_total ~ pilot + C(pair_id) + {cov}{extra}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["school_id"]})
    b, se = float(m.params["pilot"]), float(m.bse["pilot"])
    sensitivity.append({"spec": lab, "b": round(b, 1),
                        "ci": [round(b - 1.96 * se, 1), round(b + 1.96 * se, 1)],
                        "sigma": round(b / 100, 3), "p": float(m.pvalues["pilot"])})

# ------------------------------------------- 6. розподіл внутрішньопарних різниць
spread_chart = {r["label"]: r["pair_spread_sd"] for r in rows if r["pair_spread_sd"]}

out = {
    "note_design": ("Пари сформували автори дослідження; ідентифікатор пари прийшов "
                    "у вихідних даних. Алгоритм підбору, порядок обробки пропусків "
                    "і донорський пул закладів нам невідомі."),
    "matched_on": matched_on,
    "balance": rows,
    "blocks": BLOCKS,
    "balance_excluded": rows_excluded,
    "note_excluded": ("Історія ЗНО/НМТ до таблиці не входить: ці ознаки відомі лише для "
                      "частини закладів, причому для пілотних і непілотних — для різних. "
                      "SMD там порівнює дві неповні вибірки, а не підібрані групи."),
    "summary": summary,
    "link_to_2026": link,
    "sensitivity": sensitivity,
    "pair_spread": spread_chart,
}
(OUT / "11_balance.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

print("=== за чим збігаються пари")
for k, v in matched_on.items():
    print(f"  {k:34s} {v['same']} із {v['of']}")
print(f"\n=== баланс: |SMD| < 0,1 у {summary['n_under_0.1']} із {summary['n_vars']}; "
      f"похибка SMD при {n_arm} школах = ±{summary['ci95_smd']}")
print(f"{'характеристика':36s}{'пілот':>9s}{'непілот':>9s}{'SMD':>8s}{'розкид пар':>12s}")
for key, title in BLOCKS.items():
    print(f"— {title}")
    for r in sorted([r for r in rows if r["block"] == key], key=lambda r: -abs(r["smd"])):
        print(f"  {r['label']:34s}{r['mean_pilot']:9.3f}{r['mean_nonpilot']:9.3f}"
              f"{r['smd']:+8.3f}{(r['pair_spread_sd'] or 0):12.2f}")
print(f"\nознак тла родин на користь пілоту: "
      f"{summary['background_favor_pilot']} із {summary['background_n']}")
print("\n=== поза таблицею: історія ЗНО/НМТ (неповні й неоднакові вибірки)")
for r in rows_excluded:
    print(f"  {r['label']:34s}{r['mean_pilot']:9.3f}{r['mean_nonpilot']:9.3f}"
          f"{r['smd']:+8.3f}   шкіл {r['n_pilot']}/{r['n_nonpilot']}, повних пар {r['n_pairs']}")
print("\n=== зв'язок внутрішньопарних різниць із різницею 2026")
for v in link.values():
    print(f"  {v['label']:22s} пар={v['n_pairs']:3d}  r={v['r']:+.3f}")
print("\n=== стійкість ефекту до дисбалансу")
for s in sensitivity:
    print(f"  {s['spec']:26s} {s['b']:+6.1f} бала  [{s['ci'][0]:+.1f}; {s['ci'][1]:+.1f}]  p={s['p']:.2g}")
