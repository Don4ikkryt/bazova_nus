"""Етап 3. Історичний бейслайн ЗНО/НМТ по тих самих школах.

Питання: чи були пілотні школи сильнішими за свої школи-відповідники ЩЕ ДО реформи?

На виході:
  output/school_baseline.csv
  output/03_baseline.json
"""
import difflib
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, ROOT

THRESHOLD = 0.45   # мінімальна оцінка збігу назв закладів

YEARS = {
    "2018": dict(file="2018_points.csv", name="EONAME", region="EORegName",
                 subjects={"Українська мова та література": "UkrBall100",
                           "Математика": "mathBall100",
                           "Історія України": "histBall100"}, scale=1.0),
    "2021": dict(file="2021_points.csv", name="EONAME", region="EORegName",
                 subjects={"Українська мова та література": "UMLBall100",
                           "Математика": "MathBall100",
                           "Історія України": "HistBall100"}, scale=1.0),
    "2024": dict(file="nmt_bazova_2024.csv", name="institution_name",
                 region="institution_region",
                 subjects={"Українська мова та література": "UkrBlockBall100",
                           "Математика": "MathBlockBall100",
                           "Історія України": "HistBlockBall100"}, scale=0.1),
    "2025": dict(file="nmt_bazova_2025.csv", name="institution_name",
                 region="institution_region",
                 subjects={"Українська мова та література": "UkrBlockBall100",
                           "Математика": "MathBlockBall100",
                           "Історія України": "HistBlockBall100"}, scale=0.1),
}


def norm_name(x):
    x = unicodedata.normalize("NFKC", str(x)).lower()
    x = x.replace("«", " ").replace("»", " ").replace('"', " ").replace("’", "'")
    x = x.replace("№", " ")
    x = re.sub(r"[^\w\s']", " ", x, flags=re.UNICODE)
    # NFKC перетворює «№» на «No» — розділяємо букви й цифри, інакше
    # «№153» лишається одним токеном і номер закладу не витягується
    x = re.sub(r"(?<=[^\W\d])(?=\d)|(?<=\d)(?=[^\W\d])", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# Спільного ідентифікатора закладу немає: у ЗНО його не публікують, а в даних
# Діагностики школа записана лише назвою. Тож зіставляти можна тільки за назвою.
# Після реформи 2020 року заклади масово перейменували («Дружківська ЗОШ І-ІІІ ст.
# № 12» -> «гімназія №12 Дружківської міської ради»), тому там, де назва не
# збігається цілком, порівнюємо стійкі частини тієї самої назви: номер закладу
# та основи власних назв (топонім, ім'я патрона).
STOP = {
    "заклад", "закладу", "загальної", "середньої", "освіти", "загальноосвітня",
    "загальноосвітній", "загальноосвітнього", "школа", "школи", "ступенів",
    "ступеня", "ступені", "комунальний", "комунальна", "комунальне", "ліцей",
    "ліцею", "гімназія", "гімназії", "навчально", "виховний", "комплекс",
    "ради", "рада", "міської", "селищної", "сільської", "районної", "обласної",
    "області", "район", "районний", "опорний", "спеціалізована", "спеціалізований",
    "імені", "ім", "з", "та", "і", "ii", "iii", "iv", "І", "кз", "зош", "нвк",
    "об'єднаної", "територіальної", "громади", "вивченням", "мови", "класу",
    "української", "державна", "установа", "департамент", "відділ", "управління",
    "новий", "нова", "№", "no",
}
ROMAN = re.compile(r"^[iіvх]+$")


def stems(name):
    out = set()
    for w in name.split():
        if w in STOP or ROMAN.match(w) or w.isdigit() or len(w) < 5:
            continue
        out.add(w[:6])
    return out


def numbers(name):
    return {int(n) for n in re.findall(r"\b(\d{1,3})\b", name)}


def match_score(a, b):
    """0..1; None якщо номери закладів явно різні."""
    na, nb = numbers(a), numbers(b)
    if na and nb and not (na & nb):
        return None
    sa, sb = stems(a), stems(b)
    if not sa or not sb:
        return None
    jac = len(sa & sb) / len(sa | sb)
    if na and nb:
        bonus = 0.35                    # номери збігаються (інакше вийшли б вище)
    elif not na and not nb:
        bonus = 0.10                    # обидва без номера — сільські школи
    else:
        bonus = -0.15                   # один із номером, другий без — підозріло
    return min(1.0, jac + bonus)


def norm_region(x):
    return re.sub(r"\s+", " ", str(x)).strip().lower().replace("м.", "м. ")


students = pd.read_csv(OUT / "students.csv")
schools = students.groupby("school_id").agg(
    school_name=("school_name", "first"), region=("region", "first"),
    pair_id=("pair_id", "first"), pilot_school=("pilot_school", "first"),
    urban=("urban", "first"), n_students=("id", "size"),
    score_2026=("score_total", "mean"), grades_2026=("GRADE_ALL", "mean")).reset_index()
schools["nname"] = schools["school_name"].map(norm_name)
schools["nregion"] = schools["region"].map(norm_region)

match_log, year_frames = [], {}

for year, cfg in YEARS.items():
    raw = pd.read_csv(ROOT / cfg["file"], low_memory=False)
    raw["nname"] = raw[cfg["name"]].map(norm_name)
    raw["nregion"] = raw[cfg["region"]].map(norm_region)

    # бали учнів: z-оцінка всередині року й предмета
    zs = []
    for label, col in cfg["subjects"].items():
        v = pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False),
                          errors="coerce") * cfg["scale"]
        v = v.where(v >= 100, np.nan) if v.max() and v.max() > 50 else v
        # «не подолав поріг» -> нижня межа шкали (100), а не пропуск
        failed = pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False),
                               errors="coerce") * cfg["scale"]
        v = v.fillna(failed.where(failed == 0, np.nan).replace(0, 100.0))
        zs.append((v - v.mean()) / v.std(ddof=1))
    raw["z"] = pd.concat(zs, axis=1).mean(axis=1)
    raw = raw[raw["z"].notna()]

    src = raw.groupby(["nname", "nregion"]).agg(
        z=("z", "mean"), n=("z", "size"),
        orig=(cfg["name"], "first")).reset_index()

    # 1) точний збіг нормалізованої назви; 2) збіг за номером+основами в межах області
    exact = dict(zip(src["nname"], src.index))
    assigned, taken = {}, set()
    cands = []
    for i, row in schools.iterrows():
        if row["nname"] in exact:
            j = exact[row["nname"]]
            assigned[row["school_id"]] = (j, 1.0)
            taken.add(j)
            continue
        pool = src[src["nregion"] == row["nregion"]]
        for j, prow in pool.iterrows():
            sc = match_score(row["nname"], prow["nname"])
            if sc is not None and sc >= THRESHOLD:
                cands.append((sc, row["school_id"], j, row["school_name"], prow["orig"]))
    n_exact = len(assigned)   # скільки зійшлося за назвою цілком, до нечіткого етапу
    # жадібне зіставлення 1-до-1, найсильніші збіги першими
    for sc, sid, j, sname, oname in sorted(cands, key=lambda t: -t[0]):
        if sid in assigned or j in taken:
            continue
        assigned[sid] = (j, sc)
        taken.add(j)
        match_log.append({"year": year, "school": sname, "matched_to": oname,
                          "score": round(sc, 3)})

    col_z, col_n = {}, {}
    for sid, (j, ratio) in assigned.items():
        col_z[sid] = src.loc[j, "z"]
        col_n[sid] = src.loc[j, "n"]
    schools[f"z_{year}"] = schools["school_id"].map(col_z)
    schools[f"n_{year}"] = schools["school_id"].map(col_n)
    year_frames[year] = dict(n_rows=int(len(raw)), n_inst=int(src.shape[0]),
                             matched=int(len(assigned)), exact=n_exact,
                             fuzzy=int(len(assigned) - n_exact))

# зведений індекс «сила школи до реформи»
schools["baseline_pre"] = schools[["z_2018", "z_2021"]].mean(axis=1)
schools["baseline_all"] = schools[["z_2018", "z_2021", "z_2024", "z_2025"]].mean(axis=1)

# ------------------------------- ОСНОВНЕ: усі пілотні школи проти всіх непілотних
# Парне зіставлення вимагає даних для ОБОХ шкіл пари й тому втрачає більшість
# закладів. Загальне порівняння використовує кожну школу, для якої дані є.
import statsmodels.formula.api as smf

overall = {}
for metric in [f"z_{y}" for y in YEARS] + ["baseline_pre", "baseline_all", "score_2026"]:
    d = schools[[metric, "pilot_school", "region", "urban", "n_students"]].dropna(subset=[metric])
    a = d[d.pilot_school == 1][metric]
    b = d[d.pilot_school == 0][metric]
    if len(a) < 3 or len(b) < 3:
        overall[metric] = {"n_pilot": int(len(a)), "n_nonpilot": int(len(b))}
        continue
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b)
    sd_pool = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                      / (len(a) + len(b) - 2))
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    diff = float(a.mean() - b.mean())
    # той самий контраст із поправкою на область і тип місцевості
    m = smf.ols(f"{metric} ~ pilot_school + C(region) + C(urban)", data=d).fit()
    overall[metric] = {
        "n_pilot": int(len(a)), "n_nonpilot": int(len(b)),
        "mean_pilot": round(float(a.mean()), 3), "mean_nonpilot": round(float(b.mean()), 3),
        "diff": round(diff, 3),
        "ci": [round(diff - 1.96 * se, 3), round(diff + 1.96 * se, 3)],
        "cohen_d": round(float(diff / sd_pool), 3) if sd_pool else None,
        "t": round(float(t), 3), "p": float(p), "p_mannwhitney": float(pu),
        "adj_diff": round(float(m.params["pilot_school"]), 3),
        "adj_ci": [round(float(m.params["pilot_school"] - 1.96 * m.bse["pilot_school"]), 3),
                   round(float(m.params["pilot_school"] + 1.96 * m.bse["pilot_school"]), 3)],
        "adj_p": float(m.pvalues["pilot_school"]),
    }

# ------------------------------- ДОДАТКОВО: порівняння всередині пар
pairs = schools.pivot_table(index="pair_id", columns="pilot_school",
                            values=[f"z_{y}" for y in YEARS] +
                                   ["baseline_pre", "baseline_all", "score_2026"],
                            aggfunc="first")
pair_tests = {}
for metric in [f"z_{y}" for y in YEARS] + ["baseline_pre", "baseline_all", "score_2026"]:
    try:
        a = pairs[(metric, 1)]
        b = pairs[(metric, 0)]
    except KeyError:
        continue
    d = (a - b).dropna()
    if len(d) < 5:
        pair_tests[metric] = {"n_pairs": int(len(d))}
        continue
    t, p = stats.ttest_rel(a.loc[d.index], b.loc[d.index])
    try:
        w, pw = stats.wilcoxon(d)
    except ValueError:
        w, pw = np.nan, np.nan
    pair_tests[metric] = {
        "n_pairs": int(len(d)),
        "mean_pilot": round(float(a.loc[d.index].mean()), 3),
        "mean_nonpilot": round(float(b.loc[d.index].mean()), 3),
        "diff": round(float(d.mean()), 3),
        "sd_diff": round(float(d.std(ddof=1)), 3),
        "ci": [round(float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))), 3),
               round(float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d))), 3)],
        "t": round(float(t), 3), "p_ttest": round(float(p), 4),
        "p_wilcoxon": None if not np.isfinite(pw) else round(float(pw), 4),
        "share_pilot_higher": round(float((d > 0).mean()), 3),
    }

# зв'язок історичної сили школи з результатом 2026
def _corr(d, a, b):
    d = d[[a, b]].dropna()
    return (round(float(np.corrcoef(d[a], d[b])[0, 1]), 3), int(len(d))) if len(d) > 4 else (None, int(len(d)))

corr = {
    "between_years": {f"{a}~{b}": _corr(schools, f"z_{a}", f"z_{b}")[0]
                      for a, b in [("2018", "2021"), ("2021", "2024"), ("2024", "2025")]},
    "baseline_pre_vs_2026": _corr(schools, "baseline_pre", "score_2026"),
    "baseline_all_vs_2026": _corr(schools, "baseline_all", "score_2026"),
    "z2025_vs_2026": _corr(schools, "z_2025", "score_2026"),
    # той самий зв'язок, але лише де обидві оцінки спираються на достатньо учнів
    "z2025_vs_2026_bigN": _corr(schools[(schools["n_2025"] >= 25) & (schools["n_students"] >= 20)],
                                "z_2025", "score_2026"),
    "baseline_all_vs_grades": _corr(schools, "baseline_all", "grades_2026"),
    "grades_vs_score_2026": _corr(schools, "grades_2026", "score_2026"),
}
# усередині областей (знімає регіональні відмінності)
dm = schools.copy()
for c in ["baseline_all", "score_2026"]:
    dm[c] = dm[c] - dm.groupby("region")[c].transform("mean")
corr["baseline_vs_2026_within_region"] = _corr(dm, "baseline_all", "score_2026")

schools.to_csv(OUT / "school_baseline.csv", index=False)
out = {
    "years": year_frames,
    "coverage": {y: int(schools[f"z_{y}"].notna().sum()) for y in YEARS},
    "coverage_pilot": {y: int(schools[(schools.pilot_school == 1)][f"z_{y}"].notna().sum())
                       for y in YEARS},
    "coverage_nonpilot": {y: int(schools[(schools.pilot_school == 0)][f"z_{y}"].notna().sum())
                          for y in YEARS},
    "overall_tests": overall,
    "pair_tests": pair_tests,
    "correlation_with_2026": corr,
    "threshold": THRESHOLD,
    "fuzzy_matches": sorted(match_log, key=lambda r: r["score"]),
    "n_fuzzy": len(match_log),
}
(OUT / "03_baseline.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
