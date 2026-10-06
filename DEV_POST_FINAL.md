---
title: FieldDay: the allergy-smart grass window
published: true
tags: devchallenge, hf26challenge
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 1: Touch Grass](https://dev.to/challenges/hacktoberfest-week1-2026-10-05)*

## What I Built

FieldDay tells allergy families the single best hour to go outside today — then gets out of the way. Parents of sneezy kids currently check three apps (weather, AQI, pollen) and guess; guess wrong and the evening is miserable. FieldDay fuses forecast + pollen + YOUR symptom history into one 0–100 score per daylight hour, explains the pick in warm parent-language, and fits in one mobile screen. Sixty seconds, then screen off — that's the whole point of Touch Grass.

![Family picnic day by the lake](https://thumb.wikimedia.org/wikipedia/commons/thumb/f/fa/Family_Picnic_Near_Orchard_Point_Marina.jpg/1280px-Family_Picnic_Near_Orchard_Point_Marina.jpg)
*The whole point: more days like this. (Rick Obst, CC BY 2.0, via Wikimedia Commons)*

![Kids playing under a blossoming tree](https://thumb.wikimedia.org/wikipedia/commons/thumb/6/62/Children_playing_under_a_blossoming_tree%2C_Central_Park%2C_NYC.jpg/1280px-Children_playing_under_a_blossoming_tree%2C_Central_Park%2C_NYC.jpg)
*Beautiful — and exactly what triggers the sneezes. (Peter Salanki, CC BY 2.0, via Wikimedia Commons)*

## Demo

**Live:** https://fieldday-r66v.onrender.com (free Render tier, PWA — installable, works from the home screen)

Tested live Oct 6, Hanoi, straight from the production API: best window 08:00 local (score 76/100, personal flare risk 0.227 via TabPFN on 48 rows), worst 35/100 later in the week. Every hour is labeled with the model that scored it. Outdoor verification with photos lands this week — updating this post.

## Code

https://github.com/memeshee/fieldday

FastAPI + vanilla JS, one-click deploy via `render.yaml`, 6 smoke tests with CI.

## How I Built It

Open-source AI is the core, not a garnish:

- **TabPFN (Prior Labs tabular foundation model)** predicts personal flare risk fitted on YOUR symptom log — not a generic county pollen number. Your browser gets a random anonymous ID, so your taps train your forecast and nobody else's. Lightweight `tabpfn-client` so Render's free tier never OOMs; self-hosters can swap in local `tabpfn` with zero code changes.
- **Gemma (open weights)** writes the coach note. Chain: local Ollama `gemma3:1b` → hosted Gemma via Gemini API → labeled template fallback. The UI always names which source wrote the note — no silent fallbacks.
- **Free keyless data:** Open-Meteo forecast + pollen, geocoding included.

Hardest part: latency. v1 called TabPFN once per daylight hour — 7s × 24, guaranteed timeout. Fixed by fitting once and batch-predicting all hours in one call, then caching the predictor keyed on the training-data fingerprint: 37s cold → 1.3s warm in prod. Second hardest: Open-Meteo 429-throttles Render's shared IP, which once blanked the whole demo — now retry + 30-min upstream cache + an honestly-labeled MET Norway backup feed. Third: labeling every AI output with its provenance (`risk_source`, coach source), including when it's "just" the baseline.

## Why Does Open Innovation Matter?

Symptom logs are health-adjacent — they must never train someone else's closed model or require an account. Open weights + a tabular foundation model give a personal forecast with no account, no tracking, no per-call price tag: TabPFN inference sees anonymized rows and never trains on them, Gemma runs local-first. $0 to run, any model swappable. Closed APIs would make this a privacy compromise and a billing meter; open makes it a tool parents can trust.

## My Agent Session

Built pair-programming with an AI agent (Hermes) — roughly 280 tool iterations across one session: real TabPFN verification, predictor cache, per-user logs, hosted-Gemma chain, PWA shell, upstream-resilience fallback, this post. No DevRelay link (session ran outside DevRelay); the full build log lives in the repo's commit history.

## Prize Categories

- Best Use of TabPFN
- Best Use of Gemma
- Best Use of Render
- Overall
