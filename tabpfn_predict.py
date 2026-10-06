"""Flare-risk prediction with TabPFN (Prior Labs tabular foundation model).

Trains on the user's own symptom log CSV and predicts P(symptoms>=2)
for a candidate hour. Uses the lightweight TabPFN cloud client
(tabpfn-client, no torch needed) when TABPFN_TOKEN is set; tries the
local tabpfn package as a second option; falls back to a transparent
heuristic baseline when there are fewer than 10 logged rows or no
TabPFN backend is reachable.

Privacy note: the cloud client sends your symptom rows to Prior Labs'
inference API for prediction. Rows are not used to train shared models,
no account is required, and deleting your log removes your data.
"""
import csv
import os
import re
import threading

# Get a free key: log in at https://ux.priorlabs.ai, accept the license
# (Licenses tab), copy the key from https://ux.priorlabs.ai/account, then:
#   export TABPFN_TOKEN="<key>"   (both tabpfn and tabpfn-client read it)

FEATURES = ["hour", "temp_c", "pollen_index", "wind_kph"]
TARGET_CUT = 2  # symptoms >= 2 counts as a flare
MIN_ROWS = 10

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLE = os.path.join(DATA_DIR, "symptoms_sample.csv")

_UID_RE = re.compile(r"[A-Za-z0-9-]{1,32}")

# Fitted-predictor cache: key (uid, train_fingerprint) -> (predict_fn, source).
# Warm requests skip re-fit entirely (fit is the expensive half of inference).
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def log_path(uid=None) -> str:
    """Per-user log file. Unknown/unsafe uids fall back to the shared log."""
    if uid and _UID_RE.fullmatch(str(uid)):
        return os.path.join(DATA_DIR, f"logs_{uid}.csv")
    return os.path.join(DATA_DIR, "logs.csv")


def _train_files(uid=None):
    log = log_path(uid)
    files = [log] if os.path.exists(log) else []
    if os.path.exists(SAMPLE):
        files.append(SAMPLE)
    return files


def _fingerprint(uid=None) -> str:
    """Cheap change-detector for training data: path+mtime+size per file."""
    parts = [f"tok={bool(os.environ.get('TABPFN_TOKEN'))}"]
    for p in _train_files(uid):
        try:
            st = os.stat(p)
            parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{p}:missing")
    return "|".join(parts)


def _load_rows(uid=None):
    rows = []
    for p in _train_files(uid):
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


def _heuristic(hour, temp_c, pollen_index, wind_kph):
    return min(0.9, 0.08 * pollen_index + max(0, abs(temp_c - 21) - 3) * 0.02
               + max(0, wind_kph - 15) * 0.01 + (0.1 if 5 <= hour <= 10 else 0))


def _fit_predictor(X, y):
    """Return a predict(samples)->probas function, cloud first, local fallback."""
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


def _cached_predictor(X, y, uid=None):
    """Fitted predictor, reused while training data is unchanged."""
    key = (uid or "", _fingerprint(uid))
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    fresh = _fit_predictor(X, y)
    with _CACHE_LOCK:
        _CACHE[key] = fresh
        # Bound memory: keep only recent fingerprints.
        if len(_CACHE) > 32:
            _CACHE.pop(next(iter(_CACHE)))
    return fresh


def flare_risks(samples: list, uid=None) -> list:
    """Batch version: fits once (cached while data is unchanged), predicts all.

    samples: list of (hour, temp_c, pollen_index, wind_kph).
    Returns list of {"risk", "source", "n_rows"} in the same order.
    """
    rows = _load_rows(uid)
    n = len(rows)
    if n < MIN_ROWS:
        return [{"risk": round(_heuristic(h, t, p, w), 3),
                 "source": "heuristic-baseline", "n_rows": n} for h, t, p, w in samples]
    try:
        import numpy as np
        X = np.array([[r["hour"], r["temp_c"], r["pollen_index"], r["wind_kph"]] for r in rows])
        y = np.array([r["flare"] for r in rows])
        if len(set(y.tolist())) < 2:
            raise ValueError("single class")
        predict, source = _cached_predictor(X, y, uid)
        probas = predict(samples)
        return [{"risk": round(float(p), 3), "source": source, "n_rows": n} for p in probas]
    except Exception as e:
        return [{"risk": round(_heuristic(h, t, p, w), 3),
                 "source": f"heuristic-fallback:{type(e).__name__}", "n_rows": n}
                for h, t, p, w in samples]


def flare_risk(hour: float, temp_c: float, pollen_index: float, wind_kph: float,
               uid=None) -> dict:
    return flare_risks([(hour, temp_c, pollen_index, wind_kph)], uid)[0]
