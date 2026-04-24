# GridGuard-AI: ERCOT Operations Intelligence

> A six-agent multi-agent system that monitors real-time ERCOT telemetry, physical weather risks, trained ML forecasts, and NERC regulatory thresholds to synthesize emergency-dispatch recommendations in seconds.

**Course:** ITAI 2376 - Deep Learning in Artificial Intelligence  
**Term:** Spring 2026  
**Repository name:** `Crump_GridGuard_ITAI2376`

---

## Team & Roles

| Member | Role |
| --- | --- |
| **DeMarcus Crump** | Multi-agent orchestration (CrewAI + Flask SSE dashboard); Weather Analyst; Grid Monitor; Grid Operator; LSTM load-forecaster training notebook. |
| **Yoana Cook** | Foundational data pipeline (`gridguard.db`); Random Forest stress classifier notebook; Regulatory Knowledge agent (ChromaDB RAG); Renewable Forecaster; Market Analyst. |

---

## Problem Statement & Target User

In February 2021, Winter Storm Uri hit Texas and ERCOT's control room could not process incoming meteorological, financial, and mechanical telemetry fast enough to stop the cascade. 4.5 million homes went dark, 246 people died, damages exceeded $195 B. The data was there. The **synthesis speed** was not.

**GridGuard-AI** solves the synthesis-speed bottleneck. Six specialist agents each own a single risk vector (weather, regulatory compliance, renewable output, physical grid load, market pricing, final dispatch) and analyze it in parallel. A seventh-layer Grid Operator synthesizes their reports into a Pydantic-validated JSON dispatch command anchored to ERCOT Load Shed Table thresholds.

**Target users:** ERCOT grid operators, utility dispatchers, and energy market traders who need real-time, mathematically grounded situational awareness.

---

## Project Option

**Option B - Multi-Agent System.** This matches the Midterm Blueprint (no path switch). A power grid requires six distinct expert roles; a single LLM attempting to reason simultaneously as meteorologist, lawyer, power engineer, and trader suffers severe context degradation. Role-isolated agents produce more deterministic and auditable outputs.

---

## Architecture Overview

The system uses a **Fan-Out / Fan-In Parallel Orchestration** on the CrewAI framework. Phase 1 (data-gathering agents: Weather, Regulatory, Renewable) and Phase 2 (intelligence agents: Grid Monitor, Market) execute concurrently via `async_execution=True`. Once all five reports are available, CrewAI passes them as `context` to the Phase 3 Grid Operator, which performs final synthesis and emits the JSON dispatch.

![GridGuard Architecture](architecture.png)

### Agent roster

| Agent | Role | Tools | DL/ML Model |
| --- | --- | --- | --- |
| Weather Analyst | Chief Climate Risk Officer | `fetch_texas_weather` | Groq Llama-3.3-70B (Transformer) |
| Regulatory Knowledge | Chief Compliance Officer | `search_compliance_database` | ChromaDB RAG over 28 NERC/ERCOT/FERC PDFs + MiniLM embeddings |
| Renewable Forecaster | Chief Renewable Integration Officer | `estimate_renewable_output` | Physics-based cube-law wind + solar irradiance model |
| Grid Monitor | Chief Reliability Officer | `fetch_ercot_grid_data`, `predict_expected_load` | 2-layer stacked **LSTM** + **Random Forest** stress classifier |
| Market Analyst | Chief Market Intelligence Officer | `fetch_realtime_prices` | Groq Llama-3.3-70B |
| Grid Operator | Chief Energy Trading Strategist | `save_dispatch_report` | Pydantic schema clamped to ERCOT Load Shed Table bounds |

---

## Deep Learning Connections (rubric criterion 3)

| Course module | Concept | Where it lives in GridGuard |
| --- | --- | --- |
| **Module 04 - RNNs** | LSTM gated memory for temporal forecasting | `notebooks/02_train_lstm.ipynb` trains `models/load_forecaster.h5`; Grid Monitor calls it at runtime |
| **Module 05 - Transformers** | Self-attention for multi-modal reasoning | Every agent's "brain" is Groq Llama-3.3-70B (transformer LLM) |
| **Module 05 - Embeddings + RAG** | Sentence-transformer embeddings and vector retrieval | `notebooks/04_build_chromadb.ipynb` builds `data/chroma_db/`; compliance agent semantic-searches it |
| **Tabular ML / Feature Engineering** | Random Forest ensemble on 39 engineered features | `notebooks/03_train_random_forest.ipynb` trains `models/random_forest.pkl`; Grid Monitor uses it as a second opinion |

---

## Frameworks & Tools

- **LLM provider:** Groq (Llama-3.3-70B-Versatile)
- **Agent orchestration:** CrewAI 0.100 (Fan-Out / Fan-In via `async_execution`)
- **Vector store / RAG:** ChromaDB 0.4.24 + `sentence-transformers/all-MiniLM-L6-v2`
- **Deep learning:** TensorFlow / Keras 2.16 (LSTM)
- **Classical ML:** scikit-learn 1.4 (Random Forest)
- **Data engineering:** pandas, numpy, sqlite3 over `data/gridguard.db` (63,351 rows, 7+ years of ERCOT telemetry joined with Open-Meteo weather)
- **Web dashboard:** Flask + Server-Sent Events (live agent-reasoning stream)
- **Live APIs:** Open-Meteo (free, no key); GridStatus.io (ERCOT load + pricing); public ERCOT endpoints

---

## Setup & Installation

### 1. Python environment
Python **3.12** is required (TensorFlow 2.16 does not yet support 3.13+). The easiest path is `pyenv` or `conda`.

```bash
# Option A: conda
conda create -n gridguard python=3.12 -y
conda activate gridguard

# Option B: venv with system Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment variables

```bash
cp .env.example .env
# Open .env and paste your Groq API key:
#   GROQ_API_KEY=gsk_...
```

Get a free Groq key at https://console.groq.com (Llama-3.3-70B is free-tier).

### 4. Generate the trained artifacts

The repo ships with the source data (`data/gridguard.db`, 28 regulatory PDFs in `data/regulatory_docs/`) but **not** the trained models or ChromaDB vector store. You have two options:

**Option A - Google Colab (recommended, no local GPU needed):**
1. Push this repo to GitHub.
2. Open each notebook in `notebooks/` via Colab (`File -> Open notebook -> GitHub`).
3. Set the `REPO_URL` environment variable in the first cell to your GitHub clone URL.
4. Run notebooks **01 -> 02 -> 03 -> 04** in order.
5. Each notebook downloads its artifacts back to your machine via `files.download()`.
6. Drop `load_forecaster.*` and `random_forest.*` into `models/`, unzip `chroma_db.zip` into `data/`, and commit.

**Option B - Local:**
```bash
jupyter lab
# then run notebooks 01 -> 02 -> 03 -> 04 top-to-bottom
```

If you skip this step, the agents still run end-to-end but log a warning and fall back to curated keyword heuristics instead of true ML inference.

---

## How to Run the Agent

### Option 1: Flask dashboard (live streaming UI)

```bash
python main.py
# then open http://localhost:5001 and click "Initiate Fan-Out Intelligence"
```

### Option 2: Pure CLI (headless)

```bash
python -m src.orchestrator
```

### Option 3: Docker

```bash
docker build -t gridguard .
docker run -p 5001:5001 --env-file .env gridguard
```

Every run writes the final JSON dispatch to `final_dispatch.json` at repo root.

---

## Example Scenarios

### Scenario 1 - Extreme heat + reserve tightening
- **Input:** Open-Meteo reports 106 °F in Dallas. GridStatus reports 81,000 MW load.
- **Agent reasoning:** Weather flags "Extreme Heat Alert." LSTM forecasts 76,200 MW vs. observed 81,000 MW -> 6.3 % deviation, anomaly fires. RF stress probability 78 %. Compliance retrieves NERC BAL-001 reserve-margin thresholds and ERCOT EEA1 protocol.
- **Dispatch:** `{"action": "emergency_dispatch", "target_mw": 1200, "compliance_verified": true, "notes": "Reserve margin approaching BAL-001 threshold; deploying quick-start generation under ERCOT EEA1."}`

### Scenario 2 - Isolated price spike, physical grid stable
- **Input:** 72 °F, HB_South spikes to $500/MWh, physical reserves healthy.
- **Agent reasoning:** Market flags financial congestion. Grid Monitor reports normal LSTM deviation (1.2 %). Renewable Forecaster confirms high solar output.
- **Dispatch:** `{"action": "monitor", "target_mw": 0, "compliance_verified": true, "notes": "Congestion localized to HB_South trading hub; no physical dispatch required."}`

### Scenario 3 - Winter freeze + wind collapse (Uri-style)
- **Input:** 21 °F statewide, wind output drops to near-zero.
- **Agent reasoning:** Weather flags winter freeze. Renewable Forecaster reports wind collapse. Grid Monitor LSTM deviation 8.1 %, RF stress probability 94 %. Compliance RAG retrieves Uri 2021 post-mortem + EEA3 Load Shed Tables.
- **Dispatch:** `{"action": "load_shed", "target_mw": -2000, "compliance_verified": true, "notes": "EEA3 controlled outages invoked per ERCOT Nodal Protocols Section 6; lessons applied from FERC/NERC Uri 2021 post-mortem."}`

---

## Repository Layout

```
Crump_GridGuard_ITAI2376/
+-- README.md
+-- REFLECTION.md
+-- requirements.txt
+-- .env.example
+-- .gitignore
+-- architecture.png
+-- main.py                        # Flask dashboard + live SSE stream
+-- Dockerfile
+-- docs/
|   +-- MD_Blueprint_Crump_DeMarcus_Cook_Yoana_GridGuard_ITAI2376.md
+-- src/                           # Agent implementations
|   +-- orchestrator.py
|   +-- weather_agent.py
|   +-- compliance_agent.py        # ChromaDB RAG
|   +-- renewable_agent.py
|   +-- grid_monitor_agent.py      # LSTM + RF inference
|   +-- market_agent.py
|   +-- grid_operator_agent.py     # Pydantic dispatch schema
+-- notebooks/                     # Reproducible training (Colab + local)
|   +-- 01_database_exploration.ipynb
|   +-- 02_train_lstm.ipynb
|   +-- 03_train_random_forest.ipynb
|   +-- 04_build_chromadb.ipynb
+-- data/
|   +-- gridguard.db               # 63,351 rows, 77 cols, 2019-2026
|   +-- regulatory_docs/           # 28 NERC/ERCOT/FERC PDFs (~100 MB)
|   |   +-- nerc_standards/        (13 PDFs)
|   |   +-- ercot_protocols/       (4 PDFs)
|   |   +-- historical_incidents/  (4 PDFs inc. Uri 2021, Elliott 2022)
|   |   +-- ferc_regulations/      (2 PDFs)
|   |   +-- texas_regulations/     (2 PDFs)
|   |   +-- ai_governance/         (2 PDFs)
|   |   +-- doe_energy/            (1 PDF)
|   +-- chroma_db/                 # Generated by notebook 04
+-- models/                        # Generated by notebooks 02 + 03
|   +-- load_forecaster.h5
|   +-- load_forecaster_scaler.npz
|   +-- load_forecaster_config.json
|   +-- random_forest.pkl
|   +-- rf_feature_names.json
|   +-- rf_config.json
+-- demo/
    +-- Crump_GridGuard_ITAI2376_Demo.mp4   # 3-5 min demo video
```

---

## Known Limitations

- **Groq free-tier rate limits.** With six agents running concurrently, tokens-per-minute can occasionally trip a 429. Tool outputs are deliberately trimmed to minimize context length.
- **ERCOT cloud-IP blocking.** ERCOT returns HTTP 403 to most cloud egress IPs. `fetch_realtime_prices` and `fetch_ercot_grid_data` transparently fall back to the most recent `gridguard.db` snapshot, then to a deterministic random simulation.
- **LSTM training window.** The blueprint-locked LSTM trains on the last 90 days of `gridguard.db`. It is calibrated to 2026 seasonal patterns; forecast quality would degrade on pre-2019 data or on a future regime change.
- **Regulatory KB is retrieval-only.** The compliance agent surfaces the applicable standard but does not parse numeric thresholds programmatically - the LLM reasons over the retrieved passages.
- **Demo is offline.** The demo video shows the local Flask dashboard; it does not hit live Groq quotas during playback.

---

## Demo Video

The 3-5 minute demo video ships inside the repo at:

```
demo/Crump_GridGuard_ITAI2376_Demo.mp4
```

It walks through three distinct scenarios (extreme heat, isolated price spike, winter freeze) and shows the live SSE stream of every agent's ReAct reasoning loop.

---

## Reproducibility Checklist

- [x] `requirements.txt` with pinned versions
- [x] `.env.example` with all required keys (and no real secrets committed)
- [x] `.gitignore` excludes `.env`, `__pycache__`, `.ipynb_checkpoints`, `.DS_Store`
- [x] Source data (`gridguard.db`, 28 regulatory PDFs) committed
- [x] Reproducible training notebooks for every ML artifact
- [x] Architecture diagram as `architecture.png` at repo root
- [x] Graceful runtime fallbacks so the agents run even if model artifacts are missing
