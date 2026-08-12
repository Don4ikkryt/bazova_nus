"""Етап 7. Зведення всіх чисел в один файл для звіту. -> output/report_data.json"""
import json

import numpy as np
import pandas as pd

from common import OUT

data = {k: json.loads((OUT / f"{k}.json").read_text()) for k in
        ["01_meta", "02_psychometrics", "03_baseline", "04_descriptive",
         "05_effects", "06_regression", "08_matching", "10_practices",
         "11_balance", "12_psychometrics2"]}
df = pd.read_csv(OUT / "students.csv")
schools = pd.read_csv(OUT / "school_baseline.csv")

# у звіт не потрапляють назви закладів — лише розподіл якості зіставлення
fm = data["03_baseline"].pop("fuzzy_matches", [])
data["03_baseline"]["fuzzy_score_bins"] = (
    pd.Series([r["score"] for r in fm]).round(1).value_counts().sort_index().to_dict())

# дані для «карти Райта»: розподіл складності завдань і здібностей учнів
items = pd.DataFrame(data["02_psychometrics"]["items"])
theta_hist, theta_edges = np.histogram(df["theta_total"], bins=np.arange(-3.5, 3.6, 0.25))
b_hist, _ = np.histogram(items["difficulty_b"], bins=theta_edges)
data["wright"] = {"edges": theta_edges.round(3).tolist(),
                  "students": theta_hist.tolist(), "items": b_hist.tolist()}

# точки для діаграми розсіювання «бейслайн НМТ ↔ результат 2026»
sc = schools.dropna(subset=["baseline_all"])
data["scatter_baseline"] = [
    {"x": round(float(r["baseline_all"]), 3), "y": round(float(r["score_2026"]), 1),
     "pilot": int(r["pilot_school"]), "n": int(r["n_students"]),
     "region": r["region"], "urban": r["urban"]}   # без назв закладів
    for _, r in sc.iterrows()]

# бали шкіл, впорядковані (для «хребта» розкиду між школами)
sch = df.groupby(["school_id", "pilot_school"])["score_total"].agg(["mean", "size"]).reset_index()
data["school_means"] = [{"m": round(float(r["mean"]), 1), "n": int(r["size"]),
                         "pilot": int(r["pilot_school"])}
                        for _, r in sch.sort_values("mean").iterrows()]

# розподіл балів по трьох групах у вигляді часток
data["density"] = {}
edges = np.arange(220, 800, 25)
for g, sub in df.groupby("group"):
    h, _ = np.histogram(sub["score_total"], bins=edges)
    data["density"][g] = {"edges": edges.tolist(), "share": (h / h.sum()).round(4).tolist()}

(OUT / "report_data.json").write_text(json.dumps(data, ensure_ascii=False))
print("розмір:", round((OUT / "report_data.json").stat().st_size / 1024, 1), "КБ")
for k in data:
    print(" ", k)
