"""Summarize validation-only experiment runs (never reads test results)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

rows = defaultdict(list)
for rep in sorted(Path("outputs/weather_runs").glob("*/report.json")):
    r = json.loads(rep.read_text())
    assert "test" not in r, f"{rep} contains test results -- experiment runs must use --no-test-eval"
    v = r["validation"]; best = r["history"][r["best_epoch"] - 1]
    pc = v["per_class"]; bdd = v["by_source"]["bdd"]["per_class"]; dawn = v["by_source"].get("dawn", {}).get("per_class", {})
    rows[rep.parent.name.rsplit("-s", 1)[0]].append({
        "macroF1": v["macro_f1"], "acc": v["accuracy"], "bddF1": v["by_source"]["bdd"]["macro_f1"],
        "rainR": pc["rain"]["recall"], "ovcR": pc["overcast"]["recall"], "snowR": pc["snow"]["recall"],
        "bddFogR": bdd["fog"]["recall"], "dawnFogR": dawn.get("fog", {}).get("recall", float("nan")),
        "bestEp": r["best_epoch"], "trainAcc": best["train_acc"], "epochs": len(r["history"])})
keys = ["macroF1", "acc", "bddF1", "rainR", "ovcR", "snowR", "bddFogR", "dawnFogR", "trainAcc", "bestEp"]
print(f"{'config':22s}{'n':>3s}" + "".join(f"{k:>10s}" for k in keys))
for name, rs in rows.items():
    cells = []
    for k in keys:
        vals = np.array([x[k] for x in rs], dtype=float)
        cells.append(f"{vals.mean():10.3f}" if k != "bestEp" else f"{'/'.join(str(int(x)) for x in vals):>10s}")
    print(f"{name:22s}{len(rs):3d}" + "".join(cells))
print("(validation only; mean over seeds; bddFogR is over just 9 noisy-label BDD fog images)")
