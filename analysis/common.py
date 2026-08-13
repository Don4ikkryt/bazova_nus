"""Спільні утиліти: завантаження книги, нормалізація, кодування анкети."""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "Діагностика_9_Тестування_Анкетування_11_12_травня_ТВМ_БГО_2.xlsx"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

DOMAINS = {
    "1": "Мовно-літературна",
    "2": "Математична",
    "3": "Природнича",
    "4": "Громадянсько-історична",
}

GROUP_ORDER = ["Пілотний клас", "Непілотний клас у пілотній школі", "Непілотна школа"]

# Предметної розмітки в тесті немає: у книзі кожна позиція описана лише галуззю.
# Найдрібніший змістовий поділ, який дають дані, — літерний код блоку завдань
# усередині коду позиції (3.08_S253Q05Т -> блок S253 -> природничі). Дрібніше
# (біологія / фізика / хімія, алгебра / геометрія) позиції не розділяються.
SUBJECTS = {
    "read": "Читання",
    "lit": "Література",
    "math": "Математика",
    "sci": "Природничі",
    "hist": "Історія",
}

SUBJECT_OF_PREFIX = {"R": "read", "L": "lit", "PM": "math", "M": "math",
                     "S": "sci", "H": "hist"}


def subject_of(item):
    """Предметна лінія позиції за літерним кодом блоку: 1.12_L216Q13N -> lit."""
    m = re.search(r"_([A-Z]+)\d", str(item))
    if not m:
        raise ValueError(f"невідомий формат коду позиції: {item}")
    return SUBJECT_OF_PREFIX[m.group(1)]


def norm_label(s):
    """Нормалізація текстової мітки: NFKC, стиснення пробілів, нижній регістр."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("—", "—").replace("–", "–").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower().rstrip(".")
    return s


# У даних трапляються синоніми міток, яких немає в кодифікаторі, але вони
# однозначно відповідають тій самій позиції порядкової шкали.
LABEL_ALIASES = {
    "частково подібні": "здебільшого подібні",
    "частково неподібні": "здебільшого неподібні",
    "не бере участі": "не бере участі в діагностиці",
}


def latinize_code(code):
    """Коди анкети змішують кирилицю й латиницю (SQВ42 vs SQB42). Зводимо до латиниці."""
    table = str.maketrans("АВСЕНКМОРТХІ", "ABCEHKMOPTXI")
    return str(code).translate(table)


def load_book():
    xl = pd.ExcelFile(XLSX)
    return {
        "test": xl.parse("Дані тестування"),
        "anketa": xl.parse("Дані анкетування"),
        "key": xl.parse("Ключі_тестування"),
        "system": xl.parse("System data"),
        "codifier": xl.parse("Кодифікатор анкети"),
    }


def build_codifier_maps(cod):
    """{код -> {'text':..., 'map': {нормалізована мітка -> число}}}"""
    d = cod.copy()
    d.columns = ["num", "code", "text", "val", "label"]
    d["code"] = d["code"].ffill()
    d["text"] = d["text"].where(d["val"].notna()).ffill()
    d = d[d["val"].notna()]
    maps = {}
    for code, g in d.groupby("code", sort=False):
        maps[latinize_code(code)] = {
            "text": str(g["text"].iloc[0]),
            "map": {norm_label(l): float(v) for v, l in zip(g["val"], g["label"])},
        }
    return maps


def encode_column(series, code, maps):
    """Текстова відповідь -> числовий код за кодифікатором. Числа лишаємо як є."""
    code = latinize_code(code)
    entry = maps.get(code)
    if entry is None:  # напр. SAL1505 -> шукаємо шаблон родини
        family = re.sub(r"\d+$", "", code)
        cands = [k for k in maps if re.sub(r"\d+$", "", k) == family]
        entry = maps[sorted(cands)[0]] if cands else None
    if entry is None:
        return pd.to_numeric(series, errors="coerce"), 0
    m = entry["map"]
    out, unmapped = [], 0
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            out.append(float(v))
            continue
        nl = norm_label(v)
        if nl not in m and nl in LABEL_ALIASES:
            nl = LABEL_ALIASES[nl]
        if nl in m:
            out.append(m[nl])
        elif nl.replace(".", "", 1).replace("-", "", 1).isdigit():
            out.append(float(nl))
        else:
            out.append(np.nan)
            unmapped += 1
    return pd.Series(out, index=series.index, dtype=float), unmapped


def cronbach_alpha(df):
    """α Кронбаха для матриці «учні × пункти» (рядки з пропусками відкидаються)."""
    d = df.dropna()
    k = d.shape[1]
    if k < 2 or len(d) < 3:
        return np.nan
    item_var = d.var(axis=0, ddof=1).sum()
    total_var = d.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return k / (k - 1) * (1 - item_var / total_var)


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0
