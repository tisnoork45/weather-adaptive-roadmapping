"""Recompute the held-out TEST weather metrics from the frozen confusion matrix.

Source: outputs/weather_runs/final_test_eval.json (written once by outputs/weather_runs/final_eval.py,
which refuses to re-run). Two evaluation paths are stored; the reported 82.3 % / 83.3 % are the
deployed WeatherClassifier.infer() path ("weather_classifier"). Rows = true class, cols = predicted
(models._confusion_from). Accuracy, per-class P/R/F1 and macro averages are recomputed here with numpy
and compared with the stored values.
Writes data/weather_confusion_matrix.csv, data/weather_per_class_metrics.csv, data/weather_metric_check.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "outputs/weather_runs/final_test_eval.json"
OUT = Path(__file__).resolve().parents[1] / "data"
CLASSES = ["clear", "rain", "fog", "snow", "night", "overcast"]  # order of models.WeatherClassifier.classes

d = json.loads(SRC.read_text())
checks = []
for path_key in ("weather_classifier", "training_eval_code"):
    rep = d[path_key]
    assert list(rep["per_class"]) == CLASSES
    cm = np.array(rep["confusion_matrix"])
    tp = np.diag(cm).astype(float)
    prec, rec = tp / cm.sum(0), tp / cm.sum(1)
    f1 = 2 * prec * rec / (prec + rec)
    acc = tp.sum() / cm.sum()
    for name, mine, stored in [("n", cm.sum(), rep["n"]), ("accuracy", acc, rep["accuracy"]),
                               ("macro_f1", f1.mean(), rep["macro_f1"])] + \
            [(f"{c}_{m}", v, rep["per_class"][c][m2]) for i, c in enumerate(CLASSES)
             for m, m2, v in (("precision", "precision", prec[i]), ("recall", "recall", rec[i]), ("f1", "f1", f1[i]))]:
        checks.append({"path": path_key, "metric": name, "recomputed": float(mine), "stored": float(stored),
                       "abs_diff": abs(float(mine) - float(stored))})
    if path_key == "weather_classifier":
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm, index=pd.Index(CLASSES, name="true\\predicted"), columns=CLASSES).to_csv(
            OUT / "weather_confusion_matrix.csv")
        pc = pd.DataFrame({"class": CLASSES, "support": cm.sum(1), "predicted_count": cm.sum(0),
                           "precision": prec, "recall": rec, "f1": f1})
        macro = pd.DataFrame([{"class": "macro_average", "support": cm.sum(), "predicted_count": cm.sum(),
                               "precision": prec.mean(), "recall": rec.mean(), "f1": f1.mean()},
                              {"class": "accuracy", "support": cm.sum(), "predicted_count": cm.sum(),
                               "precision": np.nan, "recall": np.nan, "f1": np.nan}])
        macro.loc[1, "accuracy"] = acc
        pd.concat([pc, macro]).to_csv(OUT / "weather_per_class_metrics.csv", index=False, float_format="%.6f")
c = pd.DataFrame(checks)
c.to_csv(OUT / "weather_metric_check.csv", index=False, float_format="%.8f")
print(c[c.metric.isin(["n", "accuracy", "macro_f1"])].to_string(index=False))
print("max abs diff over all metrics:", c.abs_diff.max())
print(pd.read_csv(OUT / "weather_per_class_metrics.csv").round(4).to_string(index=False))
