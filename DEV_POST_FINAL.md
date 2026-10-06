# FieldDay: the allergy-smart grass window

FieldDay tells allergy families the single best hour to go outside today.
Entering: Overall + Best Use of TabPFN + Best Use of Gemma + Best Use of Render.

**Live:** https://fieldday-r66v.onrender.com · **Code:** https://github.com/memeshee/fieldday

![Family picnic day by the lake](https://thumb.wikimedia.org/wikipedia/commons/thumb/f/fa/Family_Picnic_Near_Orchard_Point_Marina.jpg/1280px-Family_Picnic_Near_Orchard_Point_Marina.jpg)
*The whole point: more days like this. (Rick Obst, CC BY 2.0, via Wikimedia Commons)*

## Who it's for

Parents who keep kids inside on high-pollen days because guessing wrong = a miserable sneezing evening. Current workaround: check 3 apps (weather, AQI, pollen) and guess. FieldDay fuses them into one score, personalized by your own symptom history.

![Kids playing under a blossoming tree](https://thumb.wikimedia.org/wikipedia/commons/thumb/6/62/Children_playing_under_a_blossoming_tree%2C_Central_Park%2C_NYC.jpg/1280px-Children_playing_under_a_blossoming_tree%2C_Central_Park%2C_NYC.jpg)
*Beautiful — and exactly what triggers the sneezes. (Peter Salanki, CC BY 2.0, via Wikimedia Commons)*

## How it works (60 seconds, then screen off)

1. Type your city → geocoded, free forecast + pollen data, no key, no account.
2. Every remaining daylight hour scores 0–100 (temp curve around 21°C, rain, wind, UV, storm codes) MINUS your personal flare risk × 60.
3. Flare risk comes from TabPFN, the tabular foundation model, fitted on YOUR symptom log — not a generic county pollen number. Your browser gets a random anonymous ID, so your taps train your forecast and nobody else's.
4. Gemma (open weights) explains the pick in warm plain language. Chain: local Ollama → hosted Gemma → labeled template fallback, and the UI always shows which source wrote the note — no silent fallbacks.
5. Big green card = the answer. Log symptoms in 3 taps, streak counter for every real outing. Real PWA (manifest + offline app shell), mobile-first.

## Why open beats closed here

- Symptom logs are health-adjacent. They must never train someone else's closed model or require an account. TabPFN inference sees anonymized rows (hour, temp, pollen, wind) and never trains on them; Gemma runs on open weights, local-first.
- $0 to run. Swap the model without rewriting anything.
- Fine-tunable: more logs → sharper personal boundary (e.g. "your kid flares above pollen 3 + wind 15, not the generic index").

## Tested live today

Oct 6, Hanoi, straight from the production API: best window 08:00 local (score 76/100, personal flare risk 0.227 via TabPFN on 48 rows), worst 35/100 later in the week. Every hour labeled with the model that scored it. Outdoor verification with photos lands this week — updating this post.

## Build notes

- Stack: FastAPI + vanilla JS PWA, Render free tier, tabpfn-client (lightweight, no torch), Ollama gemma3:1b local + hosted Gemma fallback, pytest smoke CI.
- Hardest part: latency. First version called TabPFN once per daylight hour — 7s × 24 hours, guaranteed timeout. Fixed by fitting once and batch-predicting all hours in one call, then caching the fitted predictor keyed on the training-data fingerprint so warm requests skip re-fit entirely (37s → 1.3s in prod). Second hardest: surviving Open-Meteo throttling Render's shared IP — retry + 30-min upstream cache + an honestly-labeled MET Norway backup feed. Third: making every AI output honestly labeled — risk_source and coach source are shown in the UI, including when it's "just" the baseline.

## What's next

Voice "go now" nudge, household profiles, fully-offline forecast cache.
