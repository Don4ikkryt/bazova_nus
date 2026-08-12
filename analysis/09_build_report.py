"""Етап 9. Вставляє дані у шаблон -> output/report.html"""
import json

from common import OUT, ROOT

tpl = (ROOT / "analysis" / "report_template.html").read_text()
data = (OUT / "report_data.json").read_text()

assert "/*__DATA__*/" in tpl, "у шаблоні немає плейсхолдера даних"
html = tpl.replace("/*__DATA__*/", data)

# сторінка не має містити персональних даних
low = html.lower()
for bad in ["@gmail", "@ukr.net", "ідентифікатор учня"]:
    assert bad not in low, f"у звіт потрапило: {bad}"

(OUT / "report.html").write_text(html)

# Артефакт публікується без обгортки — <!doctype>, <head> і <body> додає платформа.
# Для перегляду як окремої сторінки (GitHub Pages, локальний файл) потрібен повний
# документ, інакше браузер вмикає quirks mode.
# <title> і <style> зі звіту лишаються в <head>, решта йде в <body>
head_end = html.index("</style>") + len("</style>")
standalone = (
    '<!doctype html>\n<html lang="uk">\n<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    '<meta name="description" content="Діагностика 9: статистичний аналіз ефекту '
    'пілоту НУШ на навчальні результати дев\'ятикласників, травень 2026.">\n'
    '<style>*{margin:0;padding:0}</style>\n'
    + html[:head_end] +
    '\n</head>\n<body>\n' + html[head_end:] + '\n</body>\n</html>\n'
)
(OUT / "index.html").write_text(standalone)

print("готово:", OUT / "report.html", round(len(html) / 1024, 1), "КБ (для артефакта)")
print("готово:", OUT / "index.html", round(len(standalone) / 1024, 1), "КБ (окрема сторінка)")
