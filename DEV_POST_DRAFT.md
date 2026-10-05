# FieldDay: the allergy-smart grass window (DEV post draft)

Tags: devchallenge, hf26challenge, opensource, ai

## What I built (+ prize categories)
FieldDay tells allergy families the single best hour to go outside today.
Entering: Overall + Best Use of TabPFN + Best Use of Gemma + Best Use of Render.
Live: <RENDER URL> · Code: github.com/memeshee/fieldday

## Who it's for
Parents who keep kids inside on high-pollen days because guessing wrong = a
miserable sneezing evening. Current workaround: check 3 apps (weather, AQI,
pollen) and guess. FieldDay fuses them into one score, personalized by your
own symptom history.

## How it works (60 seconds, then screen off)
1. Type your city → geocoded, free Open-Meteo forecast + pollen, no key, no account.
2. Every remaining daylight hour scores 0–100 (temp curve around 21°C, rain,
   wind, UV, storm codes) MINUS your personal flare risk × 60.
3. Flare risk comes from TabPFN, the open tabular foundation model, trained on
   YOUR symptom log CSV — not a generic county pollen number.
4. Gemma 3 1B running locally via Ollama explains the pick in warm plain
   language ("Thursday 9am scores 82 — mild, calm, and your risk is only 0.1").
5. Big green card = the answer. Log symptoms in 3 taps, streak counter for
   every real outing. PWA, mobile-first.

## Why open beats closed here (required section)
- Symptom logs are health-adjacent. They should never train someone else's
  cloud model. Everything personal runs locally: TabPFN on-device, Gemma
  on-device, log CSV on-device.
- $0 to run. Swap the model (Gemma 2B/4B, Llama) without rewriting anything.
- Fine-tunable: more logs → sharper personal boundary (e.g. "your kid flares
  above pollen 3 + wind 15, not the generic index").

## I took it outside (field test + photos)
- <DATE>: <CITY> — went at <BEST HOUR>, score <X>, risk <Y>. Result: <...>
- <DATE>: control — went at a low-score hour. Result: <...>
- [photos: phone screenshot at park + grass selfie]

## Build notes / agent session
- Stack: FastAPI + vanilla JS PWA, Render free tier, TabPFN 2.x, Ollama gemma3:1b.
- DevRelay session: <LINK>
- Hardest part: <...>

## What's next
Voice "go now" nudge (ElevenLabs), household profiles, offline-cached forecasts.
