"""Етап 6. Регресія чинників на бал тестування.

З головної моделі виключено те, що вимірює сам результат (річна оцінка),
виміряне вже після тесту й про сам тест (зусилля, схожість завдань) та оцінку
учнем власних уроків — останнє ще й є передбачуваним механізмом реформи, тож
контроль на нього «з'їдав» би канал впливу пілоту.

Поведінкові ознаки учня (репетитори, домашні завдання, пропущені уроки,
тривожність, читання, екранний час, ШІ) в моделі ЛИШАЮТЬСЯ. У них невизначений
напрямок зв'язку, але вони не вимірюють результат і не зміщують інших
коефіцієнтів — перевірено: повернення змінює коефіцієнт при пілоті на 0,007.

Плюс порівняння практик викладання між трьома підгрупами.
-> output/06_regression.json
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import OUT, GROUP_ORDER

df = pd.read_csv(OUT / "students.csv")
Y = "score_total"

LABELS = {
    "pilot_class": "Навчання в пілотному класі НУШ",
    "pilot_school_only": "Непілотний клас у пілотній школі",
    "female": "Дівчина",
    "sen": "Особливі освітні потреби",
    "idp": "Статус ВПО",
    "abroad": "Перебуває за кордоном",
    "remote": "Дистанційний формат навчання",
    "online_test": "Тестування онлайн",
    "rural": "Сільська школа",
    "gymnasium": "Гімназія (проти ліцею)",
    "books": "Книг удома",
    "HOME": "Домашні ресурси",
    "parent_edu": "Освіта батьків",
    "ladder": "Самооцінка достатку родини",
    "UKRLANG": "Українська у щоденному спілкуванні",
    "WAR": "Кількість обставин війни, що заважали",
    "GRADE_ALL": "Середня річна оцінка",
    "effort": "Зусилля на тестуванні",
    "effort_if_graded": "Зусилля, якби оцінювали",
    "ANXI": "Навчальна тривожність",
    "INSTR": "Зрозумілість пояснень учителя",
    "RELAT": "Довірливі взаємини з учителями",
    "ENCO": "Діяльнісні практики на уроках",
    "TECH": "Цифрові інструменти на уроках",
    "FORM": "Формувальне оцінювання",
    "MALP": "Оцінка не за навчання",
    "AIUSE": "Використання ШІ",
    "SCREEN": "Розважальний екранний час",
    "READ": "Читацька залученість",
    "ENGAGE": "Уроки цікаві й залучають",
    "MISSED": "Пропущені уроки",
    "tutoring": "Репетитори / курси",
    "homework": "Час на домашні завдання",
    "school_cond": "Стан приміщення школи",
    "TASKSIM": "Завдання схожі на шкільні",
}

# Головна модель: усе, що визначено до тестування й не залежить від його результату
EXOG = ["pilot_class", "pilot_school_only", "female", "sen", "idp", "abroad",
        "remote", "online_test", "rural", "gymnasium", "school_cond",
        "books", "HOME", "parent_edu", "ladder", "UKRLANG", "WAR",
        # поведінка учня: напрямок зв'язку невизначений, але результату не вимірює
        "tutoring", "homework", "MISSED", "ANXI", "READ", "SCREEN", "AIUSE"]
BEHAVIOUR = {"tutoring", "homework", "MISSED", "ANXI", "READ", "SCREEN", "AIUSE"}

# Виключені як ендогенні; лишаються тільки в перевірочній моделі
WHY = {
    "outcome": ("Той самий результат іншими словами",
                "Оцінка й тест міряють ті самі знання. Сама лише річна оцінка додає 13 в. п. "
                "до пояснювальної сили моделі — але це не пояснення результату."),
    "post": ("Виміряно після тесту й про сам тест",
             "Відповідь дано, коли бал уже сформовано, тож вплинути на нього вона не могла."),
    "perception": ("Оцінка учнем власних уроків",
                   "Учень, якому предмет дається легко, оцінює ті самі уроки інакше, "
                   "ніж той, кому важко."),
}
ENDO = {
    "GRADE_ALL": "outcome",
    "effort": "post", "effort_if_graded": "post", "TASKSIM": "post",
    "INSTR": "perception", "RELAT": "perception", "ENCO": "perception",
    "TECH": "perception", "FORM": "perception", "MALP": "perception",
    "ENGAGE": "perception",
}
ENDO_LIST = list(ENDO)
BINARY = {"pilot_class", "pilot_school_only", "female", "sen", "idp", "abroad",
          "online_test", "rural", "gymnasium"}

d = df.copy()
d["pilot_class"] = (d["group"] == GROUP_ORDER[0]).astype(float)
d["pilot_school_only"] = (d["group"] == GROUP_ORDER[1]).astype(float)
d["gymnasium"] = (d["school_type"] == "гімназія").astype(float)

ALL_TERMS = EXOG + ENDO_LIST

miss = {}
for c in ALL_TERMS:
    miss[c] = round(float(d[c].isna().mean()), 3)
    if d[c].isna().mean() > 0.02:
        d[c + "_m"] = d[c].isna().astype(float)
    d[c] = d[c].fillna(d[c].median())

# стандартизація: неперервні -> z, бінарні лишаємо 0/1, вихід -> z
sd_y = d[Y].std(ddof=1)
d["_y"] = (d[Y] - d[Y].mean()) / sd_y
for c in ALL_TERMS:
    if c not in BINARY:
        s = d[c].std(ddof=1)
        d[c] = (d[c] - d[c].mean()) / s if s > 0 else 0.0


def run(terms, tag, with_region=True):
    # індикатори пропусків тільки для тих чинників, що є в моделі
    mt = [c + "_m" for c in terms if c + "_m" in d.columns]
    rhs = " + ".join(terms + mt) + (" + C(region)" if with_region else "")
    m = smf.ols(f"_y ~ {rhs}", data=d).fit(cov_type="cluster",
                                           cov_kwds={"groups": d["school_id"]})
    rows = []
    for t in terms:
        b, se = m.params[t], m.bse[t]
        rows.append({
            "term": t, "label": LABELS.get(t, t),
            "beta": round(float(b), 3), "se": round(float(se), 3),
            "ci": [round(float(b - 1.96 * se), 3), round(float(b + 1.96 * se), 3)],
            "p": float(m.pvalues[t]),
            "points": round(float(b * sd_y), 1),
            "endogenous": t in ENDO,
            "behaviour": t in BEHAVIOUR,
        })
    rows.sort(key=lambda r: -abs(r["beta"]))
    return m, {"tag": tag, "n": int(m.nobs), "k": len(terms),
               "r2": round(float(m.rsquared), 3),
               "r2_adj": round(float(m.rsquared_adj), 3), "terms": rows}


mM, MAIN = run(EXOG, "Чинники, визначені до тестування")
mF, FULL = run(ALL_TERMS, "Перевірка: з ендогенними чинниками")

# чи змінюються коефіцієнти при пілоті, коли додати ендогенні чинники
def coef(model, term):
    r = next(x for x in model["terms"] if x["term"] == term)
    return {"beta": r["beta"], "points": r["points"], "p": r["p"], "ci": r["ci"]}


pilot_stability = {t: {"main": coef(MAIN, t), "full": coef(FULL, t)}
                   for t in ("pilot_class", "pilot_school_only")}

# приріст R² по блоках
SES = ["books", "HOME", "parent_edu", "ladder"]
blocks = {
    "Лише соціально-економічні ресурси родини": round(
        float(smf.ols("_y ~ " + " + ".join(SES + [c + "_m" for c in SES if c + "_m" in d]),
                      data=d).fit().rsquared), 3),
    "Лише школа (100 закладів)": round(
        float(smf.ols("_y ~ C(school_id)", data=d).fit().rsquared), 3),
    "Регіон + тип і місцевість закладу": round(
        float(smf.ols("_y ~ C(region) + rural + gymnasium", data=d).fit().rsquared), 3),
    f"Головна модель — {len(EXOG)} чинників, визначених до тесту": round(float(MAIN["r2"]), 3),
    "Те саме + річна оцінка та інші ендогенні чинники": round(float(FULL["r2"]), 3),
}

# скільки з R² повної моделі дає сама лише річна оцінка
_, only_grade = run(EXOG + ["GRADE_ALL"], "перевірка: +річна оцінка")
r2_grade_alone = round(float(only_grade["r2"] - MAIN["r2"]), 3)

# VIF (мультиколінеарність) головної моделі
Xm = d[EXOG].astype(float).values
vif = {c: round(float(variance_inflation_factor(Xm, i)), 2) for i, c in enumerate(EXOG)}

excluded = [{"key": k, "short": WHY[k][0], "why": WHY[k][1],
             "terms": [LABELS.get(t, t) for t, kk in ENDO.items() if kk == k]}
            for k in WHY]

# ------------------------------------- практики: чи відрізняються між групами?
practice_cols = ["ENCO", "TECH", "FORM", "INSTR", "RELAT", "ENGAGE", "MALP",
                 "TASKSIM", "ANXI", "DIGI", "AIUSE", "READ", "effort",
                 "GRADE_ALL", "books", "HOME", "UKRLANG", "MISSED", "tutoring"]
practices = {}
for c in practice_cols:
    sub = df[[c, "group", "school_id", "pilot_school", "class_pilot"]].dropna()
    s = sub[c].std(ddof=1)
    means = {g: round(float(x[c].mean()), 3) for g, x in sub.groupby("group")}
    # різниця «пілотний клас − непілотний клас тієї самої школи», школа фіксована
    inner = sub[sub["pilot_school"] == 1].copy()
    both = inner.groupby("school_id")["class_pilot"].nunique().eq(2)
    inner = inner[inner["school_id"].isin(both[both].index)].copy()
    inner["pilot"] = (inner["class_pilot"] == "Пілотний").astype(float)
    m = smf.ols(f"{c} ~ pilot + C(school_id)", data=inner).fit(
        cov_type="cluster", cov_kwds={"groups": inner["school_id"]})
    # різниця «пілотна школа − непілотна школа», пара фіксована
    outer = sub.copy()
    outer["ps"] = outer["pilot_school"].astype(float)
    outer = outer.merge(df[["id", "pair_id"]].rename(columns={"id": "_id"}),
                        left_index=True, right_index=True, how="left") \
        if False else outer.join(df["pair_id"])
    m2 = smf.ols(f"{c} ~ ps + C(pair_id)", data=outer).fit(
        cov_type="cluster", cov_kwds={"groups": outer["school_id"]})
    practices[c] = {
        "label": LABELS.get(c, c),
        "means": {g: means.get(g) for g in GROUP_ORDER},
        "within_school_diff_sd": round(float(m.params["pilot"] / s), 3),
        "within_school_p": float(m.pvalues["pilot"]),
        "school_level_diff_sd": round(float(m2.params["ps"] / s), 3),
        "school_level_p": float(m2.pvalues["ps"]),
    }

out = {
    "model_main": MAIN, "model_full": FULL,
    "excluded": excluded, "n_excluded": len(ENDO),
    "pilot_stability": pilot_stability,
    "r2_from_grade_alone": r2_grade_alone,
    "r2_blocks": blocks,
    "missing_share": miss,
    "vif_max": max(vif.values()), "vif": vif,
    "sd_outcome_points": round(float(sd_y), 1),
    "practices": practices,
}
(OUT / "06_regression.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

print("=== ГОЛОВНА модель: %d чинників, R²=%.3f, n=%d" % (MAIN["k"], MAIN["r2"], MAIN["n"]))
for r in MAIN["terms"]:
    print(f"  {r['label']:42s} β={r['beta']:+.3f}  p={r['p']:.3g}")
print("=== ПЕРЕВІРКА з ендогенними: %d чинників, R²=%.3f" % (FULL["k"], FULL["r2"]))
for r in FULL["terms"][:10]:
    print(f"  {r['label']:42s} β={r['beta']:+.3f}  p={r['p']:.3g}")
print("--- стабільність коефіцієнтів при пілоті")
for t, v in pilot_stability.items():
    print(f"  {LABELS[t]:34s} головна {v['main']['beta']:+.3f}"
          f"   з ендогенними {v['full']['beta']:+.3f}")
print("--- сама лише річна оцінка додає до R²:", r2_grade_alone)
print("=== Практики: усередині школи vs між школами (в SD)")
for c, v in practices.items():
    print(f"  {v['label']:38s} внутр={v['within_school_diff_sd']:+.2f} (p={v['within_school_p']:.3g})"
          f"   міжшкіл={v['school_level_diff_sd']:+.2f} (p={v['school_level_p']:.3g})")
print("VIF max:", out["vif_max"])
