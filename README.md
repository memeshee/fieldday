# FieldDay 🌿 — allergy-smart grass window

Hacktoberfest Week 1: Touch Grass submission. Open-weight AI at the core:
- **TabPFN** (Prior Labs tabular foundation model) predicts your personal
  flare risk from your own symptom log — the forecast gets personal.
  Inference runs through Prior Labs' API on anonymized symptom rows
  (hour, temp, pollen, wind, flare yes/no); rows are never used to train
  shared models, no account needed. Self-hosters can swap in the local
  `tabpfn` package (needs torch) with zero code changes.
- **Gemma 3 1B** writes the coach note in plain parent-language.
  Chain: local Ollama → hosted Gemma (HF Inference / Groq, optional keys)
  → labeled template fallback. The UI always shows which source wrote the note.
- Free, keyless weather + pollen data (Open-Meteo). Real PWA
  (manifest + offline app-shell service worker), mobile-first:
  one screen, then go outside.

Per-visitor log files: every browser gets a random anonymous ID, so your
taps train your forecast — judges clicking around never pollute each other.

## Run locally
```
pip install -r requirements.txt
export TABPFN_TOKEN="<priorlabs key>"   # free at https://ux.priorlabs.ai/account
uvicorn app:app --port 8000
# optional: ollama pull gemma3:1b && ollama serve  (local coach notes)
# optional: export HF_TOKEN=... (or GROQ_API_KEY=...) for hosted Gemma notes
```

## Prize categories entered
- Best Use of TabPFN (featured)
- Best Use of Gemma (featured)
- Best Use of Render (featured, via render.yaml)

## Why open beats closed here
Symptom logs are health-adjacent and private — they should never train someone
else's closed model or require an account. Open weights + a tabular foundation
model = personal forecast with no account, no tracking, no per-call price tag,
and every AI output labeled with the model that produced it.

## Seed data
`data/symptoms_sample.csv` is 48 clearly-synthetic starter rows (fixed-seed
generator, ~1/3 flare days) so TabPFN has a real decision boundary from the
first click. Your own logs are appended per-visitor and take over as they grow.
