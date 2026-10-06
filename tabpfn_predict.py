"""Flare-risk prediction with TabPFN (Prior Labs tabular foundation model).

Trains on the user's own symptom log CSV and predicts P(symptoms>=2)
for a candidate hour. Uses the lightweight TabPFN cloud client
(tabpfn-client, no torch needed) when TABPFN_TOKEN is set; tries the
local tabpfn package as a second option; falls back to a transparent
heuristic baseline when there are fewer than 10 logged rows or no
TabPFN backend is reachable.
"""
import csv
import os

# Get a free key: log in at https://ux.priorlabs.ai, accept the license
# (Licenses tab), copy the key from https://ux.priorlabs.ai/account, then:
#   export TABPFN_TOKEN="<key>"   (both tabpfn and tabpfn-client read it)

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


def _predict_cloud(X, y, sample):
    """Lightweight TabPFN cloud client (no torch). Returns proba or raises."""
    from tabpfn_client import TabPFNClassifier
    clf = TabPFNClassifier()
    clf.fit(X, y)
    return float(clf.predict_proba(sample)[0][1])


def _predict_local(X, y, sample):
    """Full local tabpfn package (needs torch). Returns proba or raises."""
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(ignore_pretraining_limits=True)
    clf.fit(X, y)
    return float(clf.predict_proba(sample)[0][1])


def _heuristic(hour, temp_c, pollen_index, wind_kph):
    return min(0.9, 0.08 * pollen_index + max(0, abs(temp_c - 21) - 3) * 0.02
               + max(0, wind_kph - 15) * 0.01 + (0.1 if 5 <= hour <= 10 else 0))


def _fit_predictor(X, y):
    """Return a predict_proba(samples)->list function, cloud first, local fallback."""
    import numpy as np
    if os.environ.get("TABPFN_TOKEN"):
        try:
            from tabpfn_client import TabPFNClassifier
            clf = TabPFNClassifier()
            clf.fit(X, y)
            return lambda s: [float(p[1]) for p in clf.predict_proba(np.asarray(s))], "tabpfn"
        except Exception:
            pass  # fall through to local package
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(ignore_pretraining_limits=True)
    clf.fit(X, y)
    return lambda s: [float(p[1]) for p in clf.predict_proba(np.asarray(s))], "tabpfn"


def flare_risks(samples: list) -> list:
    """Batch version: fits once, predicts all samples in one call.

    samples: list of (hour, temp_c, pollen_index, wind_kph).
    Returns list of {"risk", "source", "n_rows"} in the same order.
    """
    rows = _load_rows()
    n = len(rows)
    if n < 10:
        return [{"risk": round(_heuristic(h, t, p, w), 3),
                 "source": "heuristic-baseline", "n_rows": n} for h, t, p, w in samples]
    try:
        import numpy as np
        X = np.array([[r["hour"], r["temp_c"], r["pollen_index"], r["wind_kph"]] for r in rows])
        y = np.array([r["flare"] for r in rows])
        if len(set(y.tolist())) < 2:
            raise ValueError("single class")
        predict, source = _fit_predictor(X, y)
        probas = predict(samples)
        return [{"risk": round(float(p), 3), "source": source, "n_rows": n} for p in probas]
    except Exception as e:
        return [{"risk": round(_heuristic(h, t, p, w), 3),
                 "source": f"heuristic-fallback:{type(e).__name__}", "n_rows": n}
                for h, t, p, w in samples]


def flare_risk(hour: float, temp_c: float, pollen_index: float, wind_kph: float) -> dict:
    return flare_risks([(hour, temp_c, pollen_index, wind_kph)])[0]
