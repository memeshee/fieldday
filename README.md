# FieldDay 🌿 — allergy-smart grass window

Hacktoberfest Week 1: Touch Grass submission. Open-source AI at the core:
- **TabPFN** (Prior Labs tabular foundation model, local) predicts your personal
  flare risk from your own symptom-log CSV — the forecast gets personal, and your
  data never leaves the device/server you control.
- **Gemma 3 1B** via Ollama (local, open-weight) writes the coach note in
  plain parent-language. Falls back to a template when Ollama isn't running.
- Free, keyless weather + pollen data (Open-Meteo). PWA, mobile-first:
  one screen, then go outside.

## Run locally
```
pip install -r requirements.txt
uvicorn app:app --port 8000
# optional: ollama pull gemma3:1b && ollama serve  (local coach notes)
```

## Prize categories entered
- Best Use of TabPFN (featured)
- Best Use of Gemma (featured)
- Best Use of Render (featured, via render.yaml)

## Why open beats closed here
Symptom logs are health-adjacent and private — they should never train someone
else's cloud model. Local open weights + local tabular FM = personal forecast
with zero data export, zero cost, works offline-capable.
