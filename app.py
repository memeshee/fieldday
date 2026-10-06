"""FieldDay — allergy-smart grass window. FastAPI backend + static PWA."""
import csv
import os
import re
import threading
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tabpfn_predict import flare_risks, log_path

app = FastAPI(title="FieldDay")
BASE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "FieldDay/1.0 (hackathon demo; github.com/memeshee/fieldday)"}
HTTP = httpx.Client(timeout=20, headers=UA)


def _get_upstream(url: str, params: dict, what: str):
    """Fetch with one retry; raise a labeled 502 instead of silent empty data."""
    last = "unknown"
    for _ in range(2):
        try:
            r = HTTP.get(url, params=params)
            if r.status_code == 200:
                return r.json()
            last = f"http-{r.status_code}"
        except Exception as e:
            last = f"{type(e).__name__}"
            time.sleep(2)
    raise HTTPException(status_code=502, detail=f"{what} unavailable ({last})")


# Upstream cache: free weather APIs throttle shared hosting IPs, so a repeat
# visit (or a second judge in the same city) must cost zero upstream calls.
# Keys round coords to ~11km; forecast TTL 30 min, geocode TTL 24 h.
_UP_CACHE = {}
_UP_LOCK = threading.Lock()


def _cached_upstream(url: str, params: dict, what: str, ttl: int, precision: int = 1):
    key = (url, tuple(sorted((k, (round(float(v), precision)
                                  if k in ("latitude", "longitude") and _is_num(v) else v))
                             for k, v in params.items())))
    now = time.time()
    with _UP_LOCK:
        hit = _UP_CACHE.get(key)
        if hit and hit[0] > now:
            return hit[1]
    data = _get_upstream(url, params, what)
    with _UP_LOCK:
        _UP_CACHE[key] = (now + ttl, data)
        if len(_UP_CACHE) > 64:
            _UP_CACHE.pop(next(iter(_UP_CACHE)))
    return data


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False

GEMMA_MODEL = "gemma3:1b"

_METNO_SYMBOLS = [  # (substring, weathercode-ish, precip_prob)
    ("thunder", 96, 90), ("hail", 75, 80), ("sleet", 67, 75),
    ("snow", 71, 75), ("rain", 63, 75), ("drizzle", 53, 45),
    ("fog", 45, 30), ("cloudy", 3, 20), ("partlycloudy", 2, 10),
    ("fair", 1, 5), ("clearsky", 0, 5),
]


def _metno_forecast(lat: float, lon: float):
    """Backup forecast via MET Norway (free, keyless) when Open-Meteo throttles.

    Returns (feats, meta) in the same shape as the primary branch.
    Precipitation probability is an approximation from symbol + amount;
    pollen falls back to a neutral index when the AQ host is also down.
    """
    data = _get_upstream(
        "https://api.met.no/weatherapi/locationforecast/2.0/complete",
        {"lat": round(lat, 4), "lon": round(lon, 4)}, "metno-forecast")
    aq = {}
    try:
        aq = _cached_upstream("https://air-quality-api.open-meteo.com/v1/air-quality", {
            "latitude": lat, "longitude": lon, "hourly": "alder_pollen,birch_pollen,grass_pollen",
            "forecast_days": 2, "timezone": "UTC"}, "pollen", ttl=1800)
    except Exception:
        pass
    aqh = aq.get("hourly", {}) or {}
    aq_times = aqh.get("time", []) or []
    feats, meta = [], []
    for entry in data.get("properties", {}).get("timeseries", []):
        try:
            ts = entry.get("time", "")
            det = entry.get("data", {}).get("instant", {}).get("details", {})
            n1 = entry.get("data", {}).get("next_1_hours") or {}
            sym = (n1.get("summary", {}).get("symbol_code", "") or "").lower()
            if "night" in sym and "day" not in sym:
                continue  # daylight hours only, mirroring the primary branch
            t = float(det.get("air_temperature", 20))
            w = float(det.get("wind_speed", 0)) * 3.6
            amt = float((n1.get("details", {}) or {}).get("precipitation_amount", 0) or 0)
            uv = float(det.get("ultraviolet_index_clear_sky", 3) or 0)
            code, prob = 2, 15
            for sub, c, p in _METNO_SYMBOLS:
                if sub in sym:
                    code, prob = c, p
                    break
            prob = min(95, max(prob, amt * 40))
            pollen_idx = 2
            if aq_times:
                try:
                    i = aq_times.index(ts[:13] + ":00" if len(ts) >= 13 else ts)
                    pk = 0.0
                    for k in ("alder_pollen", "birch_pollen", "grass_pollen"):
                        pk = max(pk, float((aqh.get(k) or [0])[i] or 0))
                    pollen_idx = min(5, int(pk // 20) if pk else 0)
                except (ValueError, IndexError):
                    pass
            score = grass_score(t, prob, w, uv, code)
            hr = int(ts[11:13])
            feats.append((hr, t, pollen_idx, w))
            meta.append({"time": ts, "hour": hr, "temp_c": round(t, 1),
                         "precip_prob": round(prob), "wind_kph": round(w, 1),
                         "uv": round(uv, 1), "pollen_index": pollen_idx,
                         "grass": score})
        except Exception:
            continue
    return feats, meta


# ---- grass scoring (own weights; daylight hours only) ----
def grass_score(temp_c, precip_prob, wind_kph, uv, code):
    s = 100.0
    s -= min(55, abs(temp_c - 21) * 3.0)
    s -= min(45, precip_prob * 0.9)
    s -= max(0, wind_kph - 14) * 1.4
    s -= max(0, uv - 5.5) * 5.0
    if code >= 95:
        s -= 55
    elif code >= 61:
        s -= 28
    elif code >= 51:
        s -= 12
    return max(0, round(s))


@app.get("/health")
def health():
    return {"ok": True, "service": "fieldday"}


@app.get("/api/geocode")
def geocode(name: str):
    return _cached_upstream("https://geocoding-api.open-meteo.com/v1/search",
                            {"name": name, "count": 5, "format": "json"},
                            "geocode", ttl=86400)


@app.get("/api/grass-window")
def grass_window(lat: float, lon: float, uid: str = ""):
    try:
        fx = _cached_upstream("https://api.open-meteo.com/v1/forecast", {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,precipitation_probability,weathercode,windspeed_10m,uv_index,is_day",
            "forecast_days": 2, "timezone": "auto"}, "forecast", ttl=1800)
        wx_source = "open-meteo"
    except HTTPException:
        # Shared hosting IPs get throttled by Open-Meteo (429). Rather than
        # dying for every judge, degrade honestly to the MET Norway feed.
        fx, wx_source = None, "metno"
    if wx_source == "metno":
        feats, meta = _metno_forecast(lat, lon)
        if not feats:
            raise HTTPException(status_code=502, detail="both weather feeds unavailable")
    else:
        aq = {}
        try:
            aq = _cached_upstream("https://air-quality-api.open-meteo.com/v1/air-quality", {
                "latitude": lat, "longitude": lon, "hourly": "alder_pollen,birch_pollen,grass_pollen",
                "forecast_days": 2, "timezone": "auto"}, "pollen", ttl=1800)
        except Exception:
            pass
        h = fx.get("hourly", {})
        n = len(h.get("time", []))
        aqh = (aq.get("hourly") or {})
    # First pass (primary feed only): collect daylight-hour features.
    if wx_source != "metno":
        feats, meta = [], []
        for i in range(n):
            try:
                if (h.get("is_day") or [1] * n)[i] != 1:
                    continue
                t = float((h.get("temperature_2m") or [20] * n)[i])
                p = float((h.get("precipitation_probability") or [0] * n)[i] or 0)
                w = float((h.get("windspeed_10m") or [0] * n)[i] or 0)
                uv = float((h.get("uv_index") or [0] * n)[i] or 0)
                code = int((h.get("weathercode") or [0] * n)[i] or 0)
                pollen = 0.0
                for k in ("alder_pollen", "birch_pollen", "grass_pollen"):
                    try:
                        v = (aqh.get(k) or [0] * n)[i]
                        pollen = max(pollen, float(v or 0))
                    except Exception:
                        pass
                pollen_idx = min(5, int(pollen // 20) if pollen else 0)
                score = grass_score(t, p, w, uv, code)
                hr = int(h["time"][i][11:13])
                feats.append((hr, t, pollen_idx, w))
                meta.append({"time": h["time"][i], "hour": hr, "temp_c": t,
                             "precip_prob": p, "wind_kph": w, "uv": uv,
                             "pollen_index": pollen_idx, "grass": score})
            except Exception:
                continue
    # One batched TabPFN call for all hours (fit once, cached while data is unchanged).
    t0 = time.time()
    risks = flare_risks(feats, uid or None) if feats else []
    inference_s = round(time.time() - t0, 1)
    hours = []
    for m, fr in zip(meta, risks):
        combined = round(max(0, min(100, m["grass"] - fr["risk"] * 60)))
        hours.append({**m, "flare_risk": fr["risk"], "risk_source": fr["source"],
                      "combined": combined})
    hours.sort(key=lambda x: -x["combined"])
    n_rows = risks[0]["n_rows"] if risks else 0
    return {"hours": hours, "best": hours[0] if hours else None,
            "risk_rows": n_rows, "inference_s": inference_s,
            "wx_source": wx_source}


class LogEntry(BaseModel):
    date: str = ""
    hour: float = 0
    temp_c: float = 25
    pollen_index: float = 2
    wind_kph: float = 8
    symptoms: float = 0
    went_out: int = 1
    uid: str = ""


@app.post("/api/log")
def log(entry: LogEntry):
    # Per-user log file: each visitor trains their own model, nobody's
    # test taps pollute anyone else's forecast.
    path = log_path(entry.uid or None)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "hour", "temp_c", "pollen_index", "wind_kph",
                        "symptoms", "went_out"])
        w.writerow([entry.date or time.strftime("%Y-%m-%d"), entry.hour,
                    entry.temp_c, entry.pollen_index, entry.wind_kph,
                    entry.symptoms, entry.went_out])
    return {"ok": True}


class ExplainReq(BaseModel):
    city: str = ""
    best: dict = {}
    risk_source: str = ""


def _gemma_prompt(city: str, b: dict) -> str:
    return (f"Write exactly 2 short sentences a parent will read on their phone. "
            f"Context: {b.get('time', 'this hour')} in {city or 'their city'} scores "
            f"{b.get('combined', '?')}/100 with personal flare risk {b.get('flare_risk', '?')} "
            f"— it is today's best hour to take the kids outside. "
            f"Warm, plain, no jargon. Output only the 2 sentences, no preamble.")


def _gemma_local(prompt: str):
    """Gemma 3 1B via Ollama on this machine (dev laptop / self-host)."""
    r = HTTP.post("http://localhost:11434/api/generate", json={
        "model": GEMMA_MODEL, "prompt": prompt, "stream": False}, timeout=60)
    if r.status_code == 200 and r.json().get("response", "").strip():
        return r.json()["response"].strip(), "gemma3-local"
    raise RuntimeError("ollama-empty")


def _gemma_hosted_hf(prompt: str):
    """Gemma 3 1B via Hugging Face Inference API (needs HF_TOKEN env)."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("no-hf-token")
    r = httpx.post("https://api-inference.huggingface.co/models/google/gemma-3-1b-it",
                   headers={"Authorization": f"Bearer {token}"},
                   json={"inputs": prompt,
                         "parameters": {"max_new_tokens": 120, "temperature": 0.7}},
                   timeout=60)
    r.raise_for_status()
    data = r.json()
    text = ""
    if isinstance(data, list) and data:
        text = (data[0].get("generated_text", "") or "").strip()
    elif isinstance(data, dict):
        text = (data.get("generated_text", "") or "").strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    if not text:
        raise RuntimeError("hf-empty")
    return text, "gemma-hosted-hf"


def _gemma_hosted_groq(prompt: str):
    """Google Gemma via Groq's hosted API (needs GROQ_API_KEY env)."""
    token = os.environ.get("GROQ_API_KEY", "")
    if not token:
        raise RuntimeError("no-groq-key")
    r = httpx.post("https://api.groq.com/openai/v1/chat/completions",
                   headers={"Authorization": f"Bearer {token}"},
                   json={"model": "gemma2-9b-it",
                         "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 150, "temperature": 0.7},
                   timeout=60)
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    if not text:
        raise RuntimeError("groq-empty")
    return text, "gemma-hosted-groq"


_NOTE_BANNED = ("`", "*", "<", ">", "?", "yes", "option", "draft",
                 "sentence", "constraint", "step", "note>", "preamble",
                 "instruction", "echo")


def _valid_note(note: str) -> bool:
    low = note.lower()
    sents = [s for s in re.split(r"[.!]\s*", note) if s.strip()]
    return (len(note) >= 40 and len(sents) >= 2
            and all(len(s) >= 15 for s in sents[:2])
            and note.count("\n") <= 2 and note.rstrip().endswith((".", "!"))
            and not any(b in low for b in _NOTE_BANNED))
def _extract_note(text: str):
    """Pull a clean 2-sentence note out of a reasoning-model reply.

    The hosted Gemma thinks out loud: instruction echoes, drafts, quoted
    options, self-checks. The finished note is consistently the LAST quoted
    string (or <note> block); validation rejects thinking fragments.
    """
    import re
    segs = text.split("<note>")
    if len(segs) > 1:
        cand = segs[-1].split("</note>")[0].strip().strip('"').strip()
        if _valid_note(cand):
            return cand
    quoted = re.findall(r'["“]([^"”]{20,400}?)["”]', text)
    for cand in reversed(quoted):
        if _valid_note(cand.strip()):
            return cand.strip()
    return ""


def _gemma_hosted_gemini(prompt: str):
    """Google Gemma via the Gemini API (needs GEMINI_API_KEY env).

    The hosted Gemma-4 reasoning model thinks out loud; _extract_note pulls
    the finished note out and anything unparsable falls through honestly.
    """
    token = os.environ.get("GEMINI_API_KEY", "")
    if not token:
        raise RuntimeError("no-gemini-key")
    last_err = "gemini-unparseable"
    for _ in range(3):  # retries: the reasoning model is nondeterministic
        r = httpx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent",
            params={"key": token},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": 600, "temperature": 0.7}},
            timeout=60)
        r.raise_for_status()
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            last_err = f"gemini-bad-shape:{str(data)[:80]}"
            continue
        note = _extract_note(text)
        if note:
            return note, "gemma-hosted-gemini"
    raise RuntimeError(last_err)


@app.post("/api/explain")
def explain(req: ExplainReq):
    b = req.best or {}
    prompt = _gemma_prompt(req.city, b)
    # Local open weights first, hosted open weights second, template last.
    # Every path is labeled honestly in `source` so the UI can show it.
    for fn in (_gemma_local, _gemma_hosted_gemini, _gemma_hosted_hf, _gemma_hosted_groq):
        try:
            text, source = fn(prompt)
            return {"text": text, "source": source}
        except Exception:
            continue
    t = (f"{b.get('time', 'This hour')} scores {b.get('combined', '?')}/100 in "
         f"{req.city or 'your city'} — mild temps, low rain odds, and your personal "
         f"flare risk is only {b.get('flare_risk', '?')}. Go now, keep it under an hour.")
    return {"text": t, "source": "template-fallback"}


app.mount("/", StaticFiles(directory=os.path.join(BASE, "static"), html=True), name="static")
