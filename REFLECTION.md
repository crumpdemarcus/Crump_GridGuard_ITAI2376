# ITAI 2376 Project Reflection

**Authors:** DeMarcus Crump & Yoana Cook  
**Project:** GridGuard-AI - Multi-Agent ERCOT Operations Intelligence

---

## What Worked Well

The Fan-Out / Fan-In orchestration on CrewAI delivered exactly what we designed it for. The Weather, Regulatory, and Renewable agents fire in parallel during Phase 1; the Grid Monitor and Market Analyst fire in parallel during Phase 2; only the final Grid Operator blocks on all five context outputs before producing the Pydantic-validated dispatch. That architecture turned a previously sequential pipeline into a parallel one without losing any reasoning quality - in our test runs, end-to-end wall-clock dropped roughly in half because Phase 1 and Phase 2 agents now overlap instead of waiting on each other.

Grounding the agents in **real data** was the second thing that worked. Rather than mocking ERCOT telemetry, we committed a 41 MB `gridguard.db` SQLite file with 63,351 rows of historical ERCOT load joined with Open-Meteo weather (2019-2026) and 28 real NERC/ERCOT/FERC regulatory PDFs. The Grid Monitor's LSTM and Random Forest both train against that real data in reproducible Colab notebooks, and the Regulatory agent retrieves from those real PDFs via ChromaDB semantic search. That grounding is what lets the Pydantic dispatch schema clamp the LLM's output to bounds pulled from actual ERCOT Load Shed Tables rather than arbitrary numbers.

Finally, the SSE-streamed Flask dashboard (`main.py`) ended up being more useful than we expected. Watching every ReAct `Thought -> Action -> Observation` cycle render live during execution made multi-agent debugging orders of magnitude easier than reading terminal logs, and it doubles as the demo surface.

---

## What Did Not Work and How We Handled It

**ERCOT blocking cloud IPs.** The ERCOT public data portal returns HTTP 403 to most cloud egress IPs, which broke every naive `fetch_realtime_prices` and `fetch_ercot_grid_data` call in our first integration tests. We solved it with tiered fallbacks: the Grid Monitor's `fetch_ercot_grid_data` tries the live `gridstatus` API first, falls back to the most recent row out of the committed `gridguard.db`, and only on a second failure emits a deterministic random simulation. The Market Analyst's `fetch_realtime_prices` is two-tier (live then simulation) because we never committed historical SPP prices to `gridguard.db` - we deliberately scoped the DB to load + weather to keep its size sane. Either way, the orchestration never deadlocks waiting on ERCOT.

**Python 3.13/3.14 vs. TensorFlow 2.16.** Our initial environment was Python 3.14, which TensorFlow does not yet support. Rather than waiting on upstream, we pinned the documented install path to Python 3.12 and re-architected the training notebooks to run primarily in Colab (free GPU, TF preinstalled, no local Python fight). The notebooks also run locally inside a 3.12 venv if the grader prefers.

**LLM hallucination of dispatch amounts.** In early tests the Grid Operator occasionally returned `target_mw` values like 12,000 MW - physically impossible for a single dispatch block. We enforce a Pydantic schema on the tool input that clamps `target_mw` to `[-5000, +5000]` MW (anchored to the ERCOT Load Shed Tables). When the LLM violates the bound, Pydantic raises a `ValidationError` which CrewAI surfaces back into the agent's next reasoning step, forcing a re-prompt with the bound explicitly stated.

---

## Biggest Technical Challenge

**Making the LSTM actually useful at runtime.** Training a 2-layer stacked LSTM on the last 90 days of `gridguard.db` was the easy part; wiring it into an agent tool that executes inside a CrewAI ReAct loop was harder. The tool has to: load the `.h5` once (not on every call), reconstruct the MinMax scaler from persisted numpy arrays, pull the last 24 hours of matching features from the DB in the exact column order the model was trained on, run inference, inverse-scale the output, and compute the deviation percentage - all in a way that gracefully no-ops and logs a warning if the model file is missing. We solved it with a lazy-loaded module-level cache (`_LSTM_BUNDLE`) plus a `sequence_length` + `features` + `scale_` + `min_` triple persisted alongside the `.h5` in `load_forecaster_config.json` and `load_forecaster_scaler.npz`. The Grid Monitor now makes one numpy array and one `.predict()` call per tool invocation.

---

## Single-Agent vs. Multi-Agent

We stayed with **Option B - Multi-Agent** (no path switch from the Midterm). A single LLM trying to simultaneously act as meteorologist, compliance lawyer, power engineer, and energy trader produced muddled reasoning in our early experiments - specifically, the single agent over-weighted whichever risk vector was most textually prominent in its input and ignored cross-cutting signals. Six role-isolated agents with narrow tool access produced dramatically more stable, auditable outputs, and the Fan-Out / Fan-In layout kept the wall-clock time competitive with the single-agent version.

---

## What We Would Build Next

1. **Live ERCOT streaming.** Today the DB is refreshed manually. Next iteration would pipe gridstatus live into `gridguard.db` via a Kafka or background worker so the LSTM always predicts on the freshest possible window.
2. **RF confidence calibration.** The Random Forest emits a raw probability. We would add `CalibratedClassifierCV` and per-hour threshold tuning so the probability is a true frequency, not a raw margin.
3. **Regulatory change detection.** Right now the ChromaDB corpus is a static snapshot. A follow-up would poll the NERC standards portal and re-embed changed documents so the compliance agent's knowledge never goes stale.
4. **Expanded DL surface.** The Renewable Forecaster is still a physics-based cube-law estimator. A learned wind/solar MW regressor (XGBoost or a small transformer) trained on historical weather-to-generation pairs would be a natural extension.
5. **Human-in-the-loop confirmation.** Before any `target_mw > 0` dispatch actually fires, a human operator should confirm via a dashboard button. The SSE stream already supports this - the plumbing just is not wired yet.
