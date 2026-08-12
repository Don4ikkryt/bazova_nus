"""Етап 10. Використання методик НУШ на уроках — за оцінками учнів.

У наборі даних немає окремої анкети для педагогів: усі свідчення про практики
викладання — це відповіді дев'ятикласників про власні уроки. Тому це показник
того, як учні бачать викладання, а не самозвіт учителів.

-> output/10_practices.json
"""
import json
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import OUT, XLSX, GROUP_ORDER, build_codifier_maps, latinize_code, norm_label

anketa = pd.read_excel(XLSX, sheet_name="Дані анкетування")
cod = pd.read_excel(XLSX, sheet_name="Кодифікатор анкети")
maps = build_codifier_maps(cod)
students = pd.read_csv(OUT / "students.csv")

BLOCKS = {
    "activity": {
        "title": "Діяльнісний і компетентнісний підхід",
        "question": "Наскільки ви погоджуєтеся з поданими твердженнями? (питання 14)",
        "items": ["SQT3801", "SQT3802", "SQT3803", "SQT3804"],
        "short": {
            "SQT3801": "Показують, як вивчене застосувати в реальному житті",
            "SQT3802": "Завдання — не просто вправи, а реальні життєві задачі",
            "SQT3803": "Практичні роботи та досліди",
            "SQT3804": "Дають змогу висловити позицію, дискутувати",
        },
    },
    "assessment": {
        "title": "Формувальне оцінювання",
        "question": "Як часто на ваших уроках у 9-му класі відбувалося таке? (питання 26)",
        "items": ["SQT5001", "SQT5002", "SQT5003", "SQT5004"],
        "short": {
            "SQT5001": "Учні самі розробляли критерії оцінювання",
            "SQT5002": "Пояснювали, за що оцінка, і радили, над чим працювати",
            "SQT5003": "Давали критерії й чеклісти для самооцінювання",
            "SQT5004": "Можна було виправити оцінку, довчивши матеріал",
        },
    },
    "malpractice": {
        "title": "Практики, яких НУШ мала позбутися",
        "question": "Як часто на ваших уроках у 9-му класі відбувалося таке? (питання 26)",
        "items": ["SQT5005", "SQT5006"],
        "short": {
            "SQT5005": "Знижували оцінку за поведінку",
            "SQT5006": "Підвищували оцінку за додаткову діяльність, а не за навчання",
        },
    },
    "digital": {
        "title": "Цифрові інструменти на уроках",
        "question": "Як часто вчителі використовували це у 9-му класі? (питання 13)",
        "items": [f"SQT37{i:02d}" for i in range(1, 7)],
        "short": {
            "SQT3701": "Інтерактивні додатки до підручників",
            "SQT3702": "Цифрові лабораторії та симуляції",
            "SQT3703": "Відеоматеріали",
            "SQT3704": "Комп'ютерні тести та квізи",
            "SQT3705": "Презентації",
            "SQT3706": "Освітні платформи",
        },
    },
    "support": {
        "title": "Підтримка та зрозумілість пояснень",
        "question": "Наскільки ви погоджуєтеся з поданими твердженнями? (питання 20)",
        "items": [f"SQT44{i:02d}" for i in range(1, 6)],
        "short": {
            "SQT4401": "Зрозуміло пояснюють, що треба робити",
            "SQT4402": "Помітили б, якби хтось був засмучений",
            "SQT4403": "Зрозуміло пояснюють матеріал",
            "SQT4404": "Чують нас і враховують нашу думку",
            "SQT4405": "Пояснюють, доки всі не зрозуміють",
        },
    },
}

# напрямок кодування: у students.csv більше = більше практики
raw_cols = {latinize_code(c): c for c in anketa.columns}
ctx = students.set_index("id")

out = {"note": "Джерело — відповіді учнів про власні уроки; окремої анкети для педагогів "
                "у наборі даних немає.", "blocks": {}}

for key, blk in BLOCKS.items():
    rows = []
    for code in blk["items"]:
        raw = anketa[raw_cols[code]]
        entry = maps.get(code)
        order = sorted(entry["map"], key=lambda k: entry["map"][k]) if entry else []
        # «важко сказати» не є точкою шкали
        order = [o for o in order if o not in ("важко сказати", "не знаю / не бажаю відповідати")]

        merged = pd.DataFrame({"id": anketa["ID-код учня"].values,
                               "raw": raw.map(norm_label).values}).set_index("id")
        merged = merged.join(ctx[["group", "school_id", "pair_id", "pilot_school",
                                  "class_pilot", code]], how="inner")
        merged = merged[merged["raw"].isin(order)]

        dist = {}
        for g in GROUP_ORDER + ["Усі"]:
            sub = merged if g == "Усі" else merged[merged["group"] == g]
            vc = sub["raw"].value_counts(normalize=True)
            dist[g] = {o: round(float(vc.get(o, 0.0) * 100), 1) for o in order}
            dist[g]["_n"] = int(len(sub))

        # частка двох «найсильніших» градацій за напрямком кодування
        v = merged[code].dropna()
        top = float(v.max())
        share_high = {}
        for g in GROUP_ORDER + ["Усі"]:
            sub = merged if g == "Усі" else merged[merged["group"] == g]
            sv = sub[code].dropna()
            share_high[g] = round(float((sv >= top - 1).mean() * 100), 1) if len(sv) else None

        sd = v.std(ddof=1)
        inner = merged[merged["pilot_school"] == 1].copy()
        both = inner.groupby("school_id")["class_pilot"].nunique().eq(2)
        inner = inner[inner["school_id"].isin(both[both].index)].copy()
        inner["pilot"] = (inner["class_pilot"] == "Пілотний").astype(float)
        m1 = smf.ols(f"{code} ~ pilot + C(school_id)", data=inner).fit(
            cov_type="cluster", cov_kwds={"groups": inner["school_id"]})
        outer = merged.copy()
        outer["ps"] = outer["pilot_school"].astype(float)
        m2 = smf.ols(f"{code} ~ ps + C(pair_id)", data=outer).fit(
            cov_type="cluster", cov_kwds={"groups": outer["school_id"]})

        rows.append({
            "code": code,
            "label": blk["short"][code],
            "full_text": entry["text"].split(" (Питання")[0] if entry else code,
            "categories": order,
            "dist": dist,
            "share_high": share_high,
            "n": int(len(merged)),
            "school_diff_sd": round(float(m2.params["ps"] / sd), 3),
            "school_p": float(m2.pvalues["ps"]),
            "within_diff_sd": round(float(m1.params["pilot"] / sd), 3),
            "within_p": float(m1.pvalues["pilot"]),
        })
    out["blocks"][key] = {"title": blk["title"], "question": blk["question"], "items": rows}

# ---------------------------------------------- зв'язок практик із результатом
link = {}
for key, blk in out["blocks"].items():
    for r in blk["items"]:
        d = students[[r["code"], "score_total", "school_id"]].dropna()
        z = (d[r["code"]] - d[r["code"]].mean()) / d[r["code"]].std(ddof=1)
        dd = d.assign(z=z)
        m = smf.ols("score_total ~ z", data=dd).fit(
            cov_type="cluster", cov_kwds={"groups": dd["school_id"]})
        link[r["code"]] = {"label": r["label"], "block": blk["title"],
                           "beta_points": round(float(m.params["z"]), 1),
                           "p": float(m.pvalues["z"])}
out["link_to_score"] = link

# скільки практик застосовується «постійно / часто» — сумарний портрет
core = [c for k in ("activity", "assessment") for c in BLOCKS[k]["items"]]
sc = students[core].dropna()
tops = {c: students[c].max() for c in core}
cnt = sum((students[c] >= tops[c] - 1).astype(float) for c in core)
students["_nush_count"] = cnt
out["intensity"] = {
    "n_items": len(core),
    "mean_by_group": {g: round(float(s["_nush_count"].mean()), 2)
                      for g, s in students.groupby("group")},
    "distribution": {g: (s["_nush_count"].value_counts(normalize=True).sort_index() * 100)
                     .round(1).to_dict() for g, s in students.groupby("group")},
}

(OUT / "10_practices.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

for k, blk in out["blocks"].items():
    print("##", blk["title"])
    for r in blk["items"]:
        print(f"  {r['label'][:52]:54s} «часто/постійно»: усі {r['share_high']['Усі']:5.1f} % | "
              f"пілот.клас {r['share_high'][GROUP_ORDER[0]]:5.1f} | непілот.школа {r['share_high'][GROUP_ORDER[2]]:5.1f} "
              f"| міжшк {r['school_diff_sd']:+.2f} (p={r['school_p']:.3g}) | внутр {r['within_diff_sd']:+.2f}")
print("\nінтенсивність (з 8 практик):", out["intensity"]["mean_by_group"])
