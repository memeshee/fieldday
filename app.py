"""FieldDay — allergy-smart grass window. FastAPI backend + static PWA."""
import csv
import os
import time

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tabpfn_predict import LOG, flare_risks

app = FastAPI(title="FieldDay")
BASE = os.path.dirname(os.path.abspath(__file__))
HTTP = httpx.Client(timeout=20)

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
    r = HTTP.get("https://geocoding-api.open-meteo.com/v1/search",
                 params={"name": name, "count": 5, "format": "json"})
    r.raise_for_status()
    return r.json()


@app.get("/api/grass-window")
def grass_window(lat: float, lon: float):
    fx = HTTP.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation_probability,weathercode,windspeed_10m,uv_index,is_day",
        "forecast_days": 2, "timezone": "auto"}).json()
    aq = {}
    try:
        aq = HTTP.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
            "latitude": lat, "longitude": lon, "hourly": "alder_pollen,birch_pollen,grass_pollen",
            "forecast_days": 2, "timezone": "auto"}).json()
    except Exception:
        pass
    h = fx.get("hourly", {})
    n = len(h.get("time", []))
    aqh = (aq.get("hourly") or {})
    # First pass: collect daylight-hour features (no model calls yet).
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
    # One batched TabPFN call for all hours (fit once, not once-per-hour).
    risks = flare_risks(feats) if feats else []
    hours = []
    for m, fr in zip(meta, risks):
        combined = round(max(0, min(100, m["grass"] - fr["risk"] * 60)))
        hours.append({**m, "flare_risk": fr["risk"], "risk_source": fr["source"],
                      "combined": combined})
    hours.sort(key=lambda x: -x["combined"])
    n_rows = risks[0]["n_rows"] if risks else 0
    return {"hours": hours, "best": hours[0] if hours else None,
            "risk_rows": n_rows}


class LogEntry(BaseModel):
    date: str = ""
    hour: float = 0
    temp_c: float = 25
    pollen_index: float = 2
    wind_kph: float = 8
    symptoms: float = 0
    went_out: int = 1


@app.post("/api/log")
def log(entry: LogEntry):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
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


@app.post("/api/explain")
def explain(req: ExplainReq):
    b = req.best or {}
    # Try local Gemma via Ollama first (open-weight, on-device).
    try:
        r = HTTP.post("http://localhost:11434/api/generate", json={
            "model": "gemma3:1b",
            "prompt": (f"You are FieldDay, a friendly outdoors coach for allergy families. "
                       f"In 2 short sentences, tell a parent why {b.get('time', 'this hour')} "
                       f"in {req.city or 'their city'} is the best hour to take the kids outside "
                       f"(score {b.get('combined', '?')}/100, flare risk {b.get('flare_risk', '?')}). "
                       f"Plain, warm, no jargon."),
            "stream": False}, timeout=60)
        if r.status_code == 200:
            return {"text": r.json().get("response", "").strip(),
                    "source": "gemma3-local"}
    except Exception:
        pass
    t = (f"{b.get('time', 'This hour')} scores {b.get('combined', '?')}/100 in "
         f"{req.city or 'your city'} — mild temps, low rain odds, and your personal "
         f"flare risk is only {b.get('flare_risk', '?')}. Go now, keep it under an hour.")
    return {"text": t, "source": "template-fallback"}


app.mount("/", StaticFiles(directory=os.path.join(BASE, "static"), html=True), name="static")
