"""Етап 1. Оцінювання тесту за ключем + збірка аналітичної таблиці учнів.

На виході:
  output/items.csv     — матриця 0/1 по 103 позиціях (лише учасники)
  output/students.csv  — один рядок на учня: групи, контекст, анкетні шкали
  output/01_meta.json  — метадані етапу
"""
import json
import re

import numpy as np
import pandas as pd

from common import (OUT, DOMAINS, GROUP_ORDER, SUBJECTS, build_codifier_maps,
                    cronbach_alpha, encode_column, latinize_code, load_book,
                    norm_label, subject_of, zscore)

book = load_book()
test, anketa, key, system = book["test"], book["anketa"], book["key"], book["system"]
maps = build_codifier_maps(book["codifier"])

# ---------------------------------------------------------------- 1. оцінювання
key.columns = ["item", "correct"]
key["item"] = key["item"].str.strip()
key["correct"] = key["correct"].map(norm_label)
key["task"] = key["item"].str.replace(r"_\d\d$", "", regex=True)
key["domain"] = key["item"].str[0]

item_cols = [c for c in test.columns if re.match(r"^\d\.\d\d_", str(c))]
assert set(item_cols) == set(key["item"]), "неузгодженість колонок і ключа"

part = test[test["Тетсування пройдено (Yes/No)"] == "Yes"].copy()
part = part.rename(columns={c: c for c in part.columns})

scored = pd.DataFrame(index=part.index)
no_answer = pd.DataFrame(index=part.index)
kmap = dict(zip(key["item"], key["correct"]))
for c in item_cols:
    given = part[c].map(norm_label)
    no_answer[c] = given.isna() | (given == "n")
    scored[c] = (given == kmap[c]).astype(int)

# ---------------------------------------------------------- 2. контекст учня
ID = "ID-код учня"
pair_col = [c for c in test.columns if "Ідентифікатор пар" in str(c)][0]

ctx = pd.DataFrame({
    "id": part[ID].values,
    "school_id": part["ID AIKOM закладу освіти"].values,
    "pair_raw": part[pair_col].values,
    "school_type_pilot": part["Пілотний / непілотний заклад "].str.strip().values,
    "class_pilot": part["Пілотний / непілотний клас"].str.strip().values,
    "region": part["Регіон, де розташований заклад освіти"].str.strip().values,
    "school_name": part["Назва закладу освіти"].str.strip().values,
    "urban": part["Тип місцевості закладу"].str.strip().values,
    "school_type": part["Тип закладу"].str.strip().values,
    "class_name": part["Назва класу"].astype(str).str.strip().values,
    "test_date": part["Дата тестування "].astype(str).str.strip().values,
}, index=part.index)

ctx["pair_id"] = ctx["pair_raw"].astype(str).str.extract(r"(\d+)")[0]
ctx["class_id"] = ctx["school_id"].astype(str) + "|" + ctx["class_name"]

def group_of(r):
    if r["school_type_pilot"].startswith("Пілотн") and r["class_pilot"] == "Пілотний":
        return GROUP_ORDER[0]
    if r["school_type_pilot"].startswith("Пілотн"):
        return GROUP_ORDER[1]
    return GROUP_ORDER[2]

ctx["group"] = ctx.apply(group_of, axis=1)
ctx["pilot_class"] = (ctx["group"] == GROUP_ORDER[0]).astype(int)
ctx["pilot_school"] = ctx["school_type_pilot"].str.startswith("Пілотн").astype(int)

# ------------------------------------- 3. бали за галузями й предметними лініями
ctx["n_items"] = len(item_cols)
ctx["raw_total"] = scored[item_cols].sum(axis=1).values
ctx["pct_total"] = 100 * ctx["raw_total"] / len(item_cols)
ctx["skipped"] = no_answer.sum(axis=1).values

for d, name in DOMAINS.items():
    cols = [c for c in item_cols if c.startswith(d + ".")]
    ctx[f"raw_d{d}"] = scored[cols].sum(axis=1).values
    ctx[f"pct_d{d}"] = 100 * ctx[f"raw_d{d}"] / len(cols)

for t, name in SUBJECTS.items():
    cols = [c for c in item_cols if subject_of(c) == t]
    ctx[f"raw_s_{t}"] = scored[cols].sum(axis=1).values
    ctx[f"pct_s_{t}"] = 100 * ctx[f"raw_s_{t}"] / len(cols)

# --------------------------------------------- 4. анкета: числове кодування
anketa_codes = [c for c in anketa.columns if str(c).startswith(("SQ", "SA", "SC"))]
enc = pd.DataFrame({"id": anketa[ID].values})
unmapped_report = {}
for c in anketa_codes:
    vals, un = encode_column(anketa[c], c, maps)
    enc[latinize_code(c)] = vals.values
    if un:
        unmapped_report[str(c)] = int(un)

# «не знаю / важко сказати» -> пропуск (не є точкою порядкової шкали)
DK = {"SQA19": 5, "SQA20": 5, "SQA24": 5, "SQB32": 6, "SQS36": 1}
for c in list(enc.columns):
    if re.match(r"^SQA22\d\d$", c) or re.match(r"^SQT44\d\d$", c):
        DK[c] = 5
for c, v in DK.items():
    if c in enc:
        enc.loc[enc[c] == v, c] = np.nan
enc["SQS36"] = enc["SQS36"] - 1              # 2..5 -> 1..4 (стан школи)
enc.loc[enc["SQH49"] == 6, "SQH49"] = 0      # «не виконував ДЗ» -> 0 годин
enc.loc[enc["SQP27"] == 4, "SQP27"] = np.nan # вік «інше»
enc["age"] = enc["SQP27"] + 13               # 1->14, 2->15, 3->16

REV = {  # шкали, де 1 = «найбільше»; переводимо у «більше = більше»
    **{f"SQF26{i:02d}": 5 for i in range(1, 4)},
    **{f"SQT37{i:02d}": 6 for i in range(1, 7)},
    **{f"SQT38{i:02d}": 5 for i in range(1, 5)},
    **{f"SQW39{i:02d}": 5 for i in range(1, 5)},
    **{f"SQW41{i:02d}": 5 for i in range(1, 5)},
    **{f"SQH43{i:02d}": 5 for i in range(1, 5)},
    **{f"SQT50{i:02d}": 5 for i in range(1, 7)},
    **{f"SQB31{i:02d}": 2 for i in range(1, 5)},   # 1=Є,2=Немає -> 1/0
    **{f"SQB30{i:02d}": 5 for i in range(1, 6)},   # -> більше = більше української
    **{f"SQA22{i:02d}": 5 for i in range(1, 6)},   # -> більше = подібніші
    "SQA19": 5, "SQA20": 5, "SQA24": 5,
}
for c, base in REV.items():
    if c in enc:
        enc[c] = base - enc[c]

# зворотно сформульовані твердження всередині вже розвернутих шкал
for c in ["SQT3802", "SQH4301", "SQH4303"]:
    enc[c] = 5 - enc[c]

# --------------------------------------------------- 5. композитні індекси
SCALES = {
    "ANXI":  (["SQF2601", "SQF2602", "SQF2603"], "Навчальна тривожність"),
    "DIGI":  (["SQW3901", "SQW3902", "SQW3903", "SQW3904"], "Цифрова впевненість"),
    "HOME":  (["SQB3101", "SQB3102", "SQB3103", "SQB3104"], "Домашні ресурси"),
    "INSTR": (["SQT4401", "SQT4403", "SQT4405"], "Зрозумілість пояснень учителя"),
    "RELAT": (["SQT4402", "SQT4404"], "Довірливі взаємини з учителями"),
    "ENCO":  (["SQT3801", "SQT3802", "SQT3803", "SQT3804"], "Діяльнісні практики на уроках"),
    "TECH":  ([f"SQT37{i:02d}" for i in range(1, 7)], "Цифрові інструменти на уроках"),
    "FORM":  (["SQT5001", "SQT5002", "SQT5003", "SQT5004"], "Формувальне оцінювання"),
    "MALP":  (["SQT5005", "SQT5006"], "Оцінка не за навчання"),
    "AIUSE": ([f"SQW41{i:02d}" for i in range(1, 5)], "Використання ШІ"),
    "SCREEN": (["SQW4001", "SQW4002", "SQW4004"], "Розважальний екранний час"),
    "READ":  (["SQH4301", "SQH4302", "SQH4303"], "Читацька залученість"),
    "ENGAGE": ([f"SQT46{i:02d}" for i in range(1, 11)], "Уроки цікаві й залучають"),
    "MISSED": ([f"SQS28{i:02d}" for i in range(1, 6)], "Пропущені уроки"),
    "UKRLANG": ([f"SQB30{i:02d}" for i in range(1, 6)], "Українська мова у спілкуванні"),
    "TASKSIM": (["SQA19", "SQA20", "SQA2201", "SQA2202", "SQA2203", "SQA2204", "SQA24"],
                "Схожість завдань на шкільні"),
}
alphas = {}
for name, (cols, label) in SCALES.items():
    cols = [c for c in cols if c in enc]
    alphas[name] = {"label": label, "k": len(cols),
                    "alpha": float(np.round(cronbach_alpha(enc[cols]), 3))
                    if cronbach_alpha(enc[cols]) == cronbach_alpha(enc[cols]) else None}
    enc[name] = enc[cols].mean(axis=1, skipna=True)

enc["WAR"] = enc[[c for c in enc.columns if c.startswith("SQB45")]].sum(axis=1, min_count=1)
enc["GRADE_L"] = enc[["SAL1501", "SAL1502", "SAL1503", "SAL1504"]].mean(axis=1)
enc["GRADE_H"] = enc[["SAH1601", "SAH1602", "SAH1603"]].mean(axis=1)
enc["GRADE_M"] = enc[["SAM1601", "SAM1602", "SAM1603"]].mean(axis=1)
enc["GRADE_S"] = enc[["SAS1701", "SAS1702", "SAS1703", "SAS1704"]].mean(axis=1)
enc["GRADE_ALL"] = enc[["GRADE_L", "GRADE_H", "GRADE_M", "GRADE_S"]].mean(axis=1)

# ----------------------------------------------------------- 6. злиття
df = ctx.merge(enc, on="id", how="left")
df["female"] = (df["SQP07"] == 2).astype(float)
df["sen"] = (df["SQP08"] > 1).astype(float)
df["abroad"] = (df["SQP09"] == 2).astype(float)
df["idp"] = (df["SQP10"] == 1).astype(float)
df["remote"] = df["SQP11"].map({1: 0.0, 2: 1.0, 3: 0.5})
df["online_test"] = (df["SQP12"] == 2).astype(float)
df["rural"] = (df["urban"] == "Сільська").astype(float)
df["books"] = df["SQB42"]
df["parent_edu"] = df["SQB32"]
df["ladder"] = df["SQB33"]
df["effort"] = df["SQE25"]
df["effort_if_graded"] = df["SQE26"]
df["difficulty"] = df[["SQD18", "SQD20", "SQD21", "SQD23"]].mean(axis=1)
df["homework"] = df["SQH49"]
df["tutoring"] = df["SQB2901"]
df["school_cond"] = df["SQS36"]

# бінарні похідні мають успадкувати пропуски вихідних питань
for src, dst in [("SQP07", "female"), ("SQP08", "sen"), ("SQP09", "abroad"),
                 ("SQP10", "idp"), ("SQP12", "online_test")]:
    df.loc[df[src].isna(), dst] = np.nan

df.to_csv(OUT / "students.csv", index=False)
scored.insert(0, "id", part[ID].values)
scored.to_csv(OUT / "items.csv", index=False)
key.to_csv(OUT / "key.csv", index=False)

meta = {
    "n_rows_total": int(len(test)),
    "n_participants": int(len(part)),
    "n_not_participating": int(len(test) - len(part)),
    "n_items": len(item_cols),
    "n_tasks": int(key["task"].nunique()),
    "n_schools": int(ctx["school_id"].nunique()),
    "n_pairs": int(ctx["pair_id"].nunique()),
    "n_classes": int(ctx["class_id"].nunique()),
    "n_regions": int(ctx["region"].nunique()),
    "groups": ctx["group"].value_counts().to_dict(),
    "items_per_domain": {DOMAINS[d]: int(sum(c.startswith(d + ".") for c in item_cols))
                         for d in DOMAINS},
    "tasks_per_domain": {DOMAINS[d]: int(key[key["domain"] == d]["task"].nunique())
                         for d in DOMAINS},
    "items_per_subject": {name: int(sum(subject_of(c) == t for c in item_cols))
                          for t, name in SUBJECTS.items()},
    "blocks_per_subject": {name: sorted({re.search(r"_([A-Z]+\d+)Q", c).group(1)
                                         for c in item_cols if subject_of(c) == t})
                           for t, name in SUBJECTS.items()},
    "scale_alphas": alphas,
    "unmapped_labels": unmapped_report,
    "anketa_missing_rate": {c: round(float(df[c].isna().mean()), 4)
                            for c in ["books", "parent_edu", "ladder", "effort", "GRADE_ALL",
                                      "ANXI", "DIGI", "HOME", "INSTR", "ENCO", "TECH", "FORM"]},
}
(OUT / "01_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

print(json.dumps(meta, ensure_ascii=False, indent=2)[:2600])
print("\nсер. % правильних:", round(df["pct_total"].mean(), 2))
print(df.groupby("group")["pct_total"].agg(["size", "mean", "std"]).round(2))
