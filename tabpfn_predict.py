"""Flare-risk prediction with TabPFN (Prior Labs tabular foundation model).

Trains on the user's own symptom log CSV and predicts P(symptoms>=2)
for a candidate hour. Falls back to a transparent heuristic baseline
when there are fewer than 10 logged rows.
"""
import csv
import os

# Free one-time setup for genuine TabPFN inference:
# 1. Log in at https://ux.priorlabs.ai, accept the license (Licenses tab)
# 2. Copy API key from https://ux.priorlabs.ai/account
# 3. export TABPFN_TOKEN="<key>"  (tabpfn reads it automatically)

FEATURES = ["hour", "temp_c", "pollen_index", "wind_kph"]
TARGET_CUT = 2  # symptoms >= 2 counts as a flare

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "symptoms_sample.csv")
LOG = os.path.join(os.path.dirname(__file__), "data", "logs.csv")


def _load_rows():
    paths = [LOG, SAMPLE] if os.path.exists(LOG) else [SAMPLE]
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({
                        "hour": float(r["hour"]),
                        "temp_c": float(r["temp_c"]),
                        "pollen_index": float(r["pollen_index"]),
                        "wind_kph": float(r["wind_kph"]),
                        "flare": 1 if float(r["symptoms"]) >= TARGET_CUT else 0,
                    })
                except (KeyError, ValueError):
                    continue
    return rows


def flare_risk(hour: float, temp_c: float, pollen_index: float, wind_kph: float) -> dict:
    rows = _load_rows()
    base = min(0.9, 0.08 * pollen_index + max(0, abs(temp_c - 21) - 3) * 0.02
               + max(0, wind_kph - 15) * 0.01 + (0.1 if 5 <= hour <= 10 else 0))
    if len(rows) < 10:
        return {"risk": round(base, 3), "source": "heuristic-baseline", "n_rows": len(rows)}
    try:
        import numpy as np
        from tabpfn import TabPFNClassifier
        X = np.array([[r["hour"], r["temp_c"], r["pollen_index"], r["wind_kph"]] for r in rows])
        y = np.array([r["flare"] for r in rows])
        if len(set(y.tolist())) < 2:
            return {"risk": round(base, 3), "source": "heuristic-baseline", "n_rows": len(rows)}
        clf = TabPFNClassifier(ignore_pretraining_limits=True)
        clf.fit(X, y)
        proba = float(clf.predict_proba(
            np.array([[hour, temp_c, pollen_index, wind_kph]]))[0][1])
        return {"risk": round(proba, 3), "source": "tabpfn", "n_rows": len(rows)}
    except Exception as e:
        return {"risk": round(base, 3), "source": f"heuristic-fallback:{type(e).__name__}",
                "n_rows": len(rows)}
