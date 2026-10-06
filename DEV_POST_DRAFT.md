# FieldDay: the allergy-smart grass window (DEV post draft)

Tags: devchallenge, hf26challenge, opensource, ai

## What I built (+ prize categories)
FieldDay tells allergy families the single best hour to go outside today.
Entering: Overall + Best Use of TabPFN + Best Use of Gemma + Best Use of Render.
Live: https://fieldday-r66v.onrender.com · Code: github.com/memeshee/fieldday

## Who it's for
Parents who keep kids inside on high-pollen days because guessing wrong = a
miserable sneezing evening. Current workaround: check 3 apps (weather, AQI,
pollen) and guess. FieldDay fuses them into one score, personalized by your
own symptom history.

## How it works (60 seconds, then screen off)
1. Type your city → geocoded, free Open-Meteo forecast + pollen, no key, no account.
2. Every remaining daylight hour scores 0–100 (temp curve around 21°C, rain,
   wind, UV, storm codes) MINUS your personal flare risk × 60.
3. Flare risk comes from TabPFN, the tabular foundation model, fitted on
   YOUR symptom log — not a generic county pollen number. Your browser gets a
   random anonymous ID, so your taps train your forecast and nobody else's.
4. Gemma 3 (open weights) explains the pick in warm plain language
   ("Thursday 9am scores 82 — mild, calm, and your risk is only 0.1").
   Chain: local Ollama → hosted Gemma → labeled template fallback, and the
   UI always shows which source wrote the note — no silent fallbacks.
5. Big green card = the answer. Log symptoms in 3 taps, streak counter for
   every real outing. Real PWA (manifest + offline app shell), mobile-first.

## Why open beats closed here (required section)
- Symptom logs are health-adjacent. They must never train someone else's
  closed model or require an account. TabPFN inference sees anonymized rows
  (hour, temp, pollen, wind) and never trains on them; Gemma runs on open
  weights, local-first.
- $0 to run. Swap the model (Gemma 2B, Llama) without rewriting anything.
- Fine-tunable: more logs → sharper personal boundary (e.g. "your kid flares
  above pollen 3 + wind 15, not the generic index").

## Images (stock, illustrative — NOT field-test evidence)
- Hero/wide: family lakeside picnic day —
  https://thumb.wikimedia.org/wikipedia/commons/thumb/f/fa/Family_Picnic_Near_Orchard_Point_Marina.jpg/1280px-Family_Picnic_Near_Orchard_Point_Marina.jpg
  (Rick Obst, CC BY 2.0, via Wikimedia Commons)
- In-post/portrait: kids under a blossoming tree, Central Park (peak pollen irony) —
  https://thumb.wikimedia.org/wikipedia/commons/thumb/6/62/Children_playing_under_a_blossoming_tree%2C_Central_Park%2C_NYC.jpg/1280px-Children_playing_under_a_blossoming_tree%2C_Central_Park%2C_NYC.jpg
  (Peter Salanki, CC BY 2.0, via Wikimedia Commons)
- Rule: these illustrate the problem/audience only. The field-test section
  below must use our own real photos, or the entry loses credibility.

## I took it outside (field test + photos)
- <DATE>: <CITY> — went at <BEST HOUR>, score <X>, risk <Y>. Result: <...>
- <DATE>: control — went at a low-score hour. Result: <...>
- [photos: phone screenshot at park + grass selfie]

## Build notes / agent session
- Stack: FastAPI + vanilla JS PWA, Render free tier, tabpfn-client (lightweight,
  no torch), Ollama gemma3:1b local + hosted Gemma fallback, pytest smoke CI.
- DevRelay session: <LINK>
- Hardest part: latency. First version called TabPFN once per daylight hour —
  7s × 24 hours, guaranteed timeout. Fixed by fitting once and batch-predicting
  all hours in one call (~15s), then caching the fitted predictor keyed on the
  training-data fingerprint so warm requests skip re-fit entirely. Second
  hardest: making every AI output honestly labeled — risk_source and coach
  source are shown in the UI, including when it's "just" the baseline.

## What's next
Voice "go now" nudge, household profiles, fully-offline forecast cache.
