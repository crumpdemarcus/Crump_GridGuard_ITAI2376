# GridGuard-AI: Intelligent Power Grid Monitoring & Dispatch System

**ITAI 2376 - Deep Learning in Artificial Intelligence**
**Midterm Blueprint - Spring 2026**
**DeMarcus Crump & Yoana Cook**

---

## 1. Problem Statement

In February 2021, Winter Storm Uri hit Texas and the power grid collapsed. Over 4.5 million homes lost electricity, nearly 250 people died, and the economic damage exceeded $195 billion. The scary part? The data was there. ERCOT - the organization that manages the Texas grid - had access to weather forecasts, power plant statuses, and demand numbers. The problem was not a lack of information. The problem was that **no one could process all of it fast enough** to stop the cascade before it started.

This is still the reality today. Grid operators sit in control rooms watching dozens of screens - weather maps, power plant dashboards, market prices, demand curves - all updating in real time. When a crisis is building, everything changes at once. A human trying to mentally connect "the wind is dying in West Texas" to "we are about to lose 1,200 MW of generation" to "pricing is spiking to $3,000/MWh" is fighting against the clock. Often, by the time the picture is clear, the window to act has already closed.

**GridGuard-AI** is a team of AI agents that does this job. It watches the weather, monitors the grid, calculates renewable displacement, checks regulatory compliance, reads the market pricing, and recommends what to do - automatically and in seconds, not minutes. The people who benefit are grid operators, utility companies, and ultimately the millions of Texans who just want to keep their lights on.

---

## 2. Option Choice

**Option B - Multi-Agent System.**

Managing a power grid is not one job - it is at least six distinct specialties. Someone needs to watch the weather. Someone needs to check the regulations. Someone needs to forecast renewable drop-offs. Someone needs to watch the physical grid load. Someone needs to watch the financial market flags. And lastly, someone needs to take all of that information and make a final call. In a real control room, these are different people with different expertise sitting in different chairs.

A single AI agent trying to do all six at once would be like asking one person to be the meteorologist, the lawyer, the engineer, the trader, and the dispatcher simultaneously. The quality drops and the reasoning gets muddied. By splitting the work across an orchestrated team of **six specialized agents** - each strictly focused on what it does best - the system mirrors how real highly-effective incident response teams operate. It is cleaner, mathematically scalable, and easier to debug when an anomaly hits.

---

## 3. Agent Architecture

The system uses a **Fan-Out / Fan-In** architecture. Instead of waiting sequentially, the agents that do not depend on each other run in parallel, maximizing speed. All six agents draw from a foundational data infrastructure - including a historical ERCOT database, a trained ML model, and a manually collected regulatory document library - built beginning October 2025. The full architecture is diagrammed below.

### Architecture Diagram

```text
+-----------------------------------------------------------------------------------+
|                       GridGuard-AI  -  Agent Architecture                         |
|              CrewAI Framework (Fan-Out / Fan-In Parallel Orchestration)           |
|                       DeMarcus Crump & Yoana Cook                                 |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|   FOUNDATIONAL DATA INFRASTRUCTURE                                                |
|   ERCOT Pipeline - SQLite DB (18,742+ records) - Random Forest Model (95.7%)     |
|   Flask API - NERC/ERCOT Regulatory Document Library (25+ documents)             |
|                                                                                   |
|   [PHASE 1: DATA INPUT]                                                           |
|   Open-Meteo API -------> [WEATHER ANALYST] -----------+                          |
|   Regulatory Doc Library -> [REGULATORY KNOWLEDGE] ----+                          |
|   Open-Meteo API -------> [RENEWABLE FORECASTER] ------+                          |
|                                                         |                         |
|   [PHASE 2: INTELLIGENCE]                               |                         |
|   GridStatus + LSTM ----> [GRID MONITOR] --------------+                          |
|   GridStatus.io --------> [MARKET ANALYST] ------------+                          |
|                                                         |                         |
|   [PHASE 3: DECISION]                                   v                         |
|                                         +--------------------------+              |
|                                         | GRID OPERATOR (DECISION) |              |
|                                         +--------------------------+              |
|                                                         |                         |
|                                                         v                         |
|                                           FINAL DISPATCH REPORT (JSON)            |
+-----------------------------------------------------------------------------------+
```

### Agent Contributions

| Agent | Role | Built By |
| --- | --- | --- |
| Weather Analyst | Chief Climate Risk Officer | DeMarcus Crump |
| Regulatory Knowledge | Chief Compliance Officer | Yoana Cook |
| Renewable Forecaster | Chief Renewable Integration Officer | Yoana Cook |
| Grid Monitor | Chief Reliability Officer | DeMarcus Crump |
| Market Analyst | Chief Market Intelligence Officer | Yoana Cook |
| Grid Operator | Chief Energy Trading Strategist | DeMarcus Crump |

### Weather Analyst
**Role: Chief Climate Risk Officer**
This agent is the "eyes on the sky." It checks real-time weather and flags conditions across Texas that pose a risk to the power grid (e.g., temperatures dropping 18F, severe storm approaching). It isolates the meteorological physical risk.

**Tool:** `fetch_texas_weather` - Open-Meteo API across 5 Texas regions (Houston, Dallas, Austin, West Texas, South Texas).

### Regulatory Knowledge
**Role: Chief Compliance Officer**
This agent ensures that all dispatch recommendations comply with NERC reliability standards and ERCOT operating protocols. It searches a manually collected regulatory document library, including the ERCOT System Operating Limit (SOL) Methodology, Summer and Winter Load Shed Tables (EEA Level 3 thresholds), Wind Integration Reports, and the FERC/NERC Uri 2021 post-mortem - using RAG-style retrieval to find the exact standard required for the current grid condition. It prevents the system from recommending actions that violate federal law.

**Tool:** `search_compliance_database` - RAG retrieval over NERC/ERCOT regulatory documents. Production implementation uses ChromaDB vector embeddings.

**Standards covered:** NERC BAL-001 (Real Power Balancing), BAL-002 (Disturbance Control), EOP-011 (Emergency Operations), IRO-006 (Reliability Coordination), ERCOT EEA Protocols (EEA1/EEA2/EEA3), FERC/NERC Uri 2021 post-mortem.

### Renewable Forecaster
**Role: Chief Renewable Integration Officer**
Since ERCOT is 40%+ renewable energy, the #1 source of supply volatility is the weather. This agent fetches live wind speed and solar irradiance data from the Open-Meteo API for 4 Texas regions and translates meteorological conditions into actionable MW generation estimates. Wind power is calculated using the physics-based cube law (power scales with wind speed cubed). Solar scales with direct radiation versus 1,000 W/m² peak. The agent spots massive drop-offs in renewable output before they hit the physical grid telemetry.

**Tool:** `estimate_renewable_output` — Open-Meteo API (West Texas, South Texas, Houston, Dallas). Calibrated against Wind Integration Reports historical baseline (2010-2016).

### Grid Monitor
**Role: Chief Reliability Officer**
While the others gather context, the Grid Monitor watches the actual power system. It pulls live ERCOT data (current total load, total capacity, reserve margin) and compares it to a trained LSTM load forecasting model to mathematically determine physical grid stress. Any deviation above 5% between the LSTM prediction and actual load is flagged as an emerging anomaly.

**Tools:** `fetch_ercot_grid_data` (GridStatus.io), `predict_expected_load` (LSTM model — load_forecaster.h5, 2.41% MAE, 24-hour sequence length).

### Price & Market Analyst
**Role: Chief Market Intelligence Officer**
This agent monitors real-time energy prices across all four major ERCOT trading hubs (HB_Houston, HB_North, HB_South, HB_West). Price spikes above $100/MWh are the earliest observable warning signal of grid distress — they appear before physical reserves formally drop. The agent uses this to ground the urgency of the dispatch recommendation.

**Tool:** `fetch_realtime_prices` — GridStatus.io ERCOT API with fallback simulation for cloud-blocked environments.

### Grid Operator
**Role: Chief Energy Trading Strategist**
This is the final decision-maker. It does not pull API data on its own; instead, it synthesizes the reports from the previous five agents. If the Renewables agent predicts a wind drop, the Grid Monitor shows tight reserves, and the Price Analyst sees spikes, the Grid Operator triggers the critical dispatch sequence with specific MW targets — constrained by the regulatory thresholds identified by the Compliance Officer.

**Tool:** `save_dispatch_report` — Pydantic-enforced JSON dispatch report generation.

---

## 4. Deep Learning Connection

### Connection 1: Transformers (Module 05)
The "brain" behind each agent is a Large Language Model built on the **Transformer architecture (Module 05)** - pulling Meta's Llama 3.3-70B via Groq LPU inference. Transformers allow these agents to reason natively about complex arrays of numerical API data. For example, connecting "dying wind in West Texas" with "plummeting temperature" into a coherent risk threat is exactly what the self-attention mechanism in Transformers was designed to weigh and optimize.

### Connection 2: Recurrent Neural Networks (RNNs / LSTMs) (Module 04)
Grid balancing is inherently a timeseries problem. The Grid Monitor relies on a **2-layer stacked LSTM** trained on 90 days of real ERCOT load history, directly drawing on our work with **Recurrent Neural Networks (Module 04)**. The model uses a 24-hour sequence length and 29,345 trainable parameters, achieving a test MAE of 2.41% of peak load — well within the 5% anomaly detection threshold. At runtime, every deviation above 5% between the LSTM prediction and actual ERCOT load flags an emerging grid anomaly before it appears in physical reserve numbers.

### Connection 3: Random Forest & Feature Engineering (Production ML)
GridGuard-AI includes a **Random Forest classifier** trained on 39 features engineered from real ERCOT telemetry — directly applying the ensemble and feature engineering concepts covered in the course. The model achieves 95.7% accuracy and 87.5% recall on stress events on a held-out temporal test set. Random Forest was chosen over neural networks for interpretability (grid operators need to understand why an alert triggered), training stability on tabular time-series data, and production deployment without GPU requirements. Top predictors: load_mw (15.8%), stress_index (13.7%), reserve_margin (12.5%), load_lag_1 (10.6%). This model informs the Grid Monitor's baseline comparisons at runtime.

### Connection 4: Vector Embeddings & RAG
The Regulatory Knowledge agent uses a locally hosted **ChromaDB vector database** seeded with a manually collected NERC/ERCOT document library. By converting multi-hundred-page federal compliance PDFs, historical blackout post-mortems, and operational limit documents into semantic embeddings, the agent uses Retrieval-Augmented Generation (RAG) to dynamically find the exact NERC standard required for the operation at hand. The documents include the ERCOT SOL Methodology, Load Shed Tables, Wind Integration Reports (2010-2016), and the FERC/NERC Uri 2021 joint investigation report.

---

## 5. Agent Framework

**Framework: CrewAI (Python)**

CrewAI was chosen to power the Fan-Out / Fan-In architecture because of its exact mapping to the way human control rooms function:
1. **Defined Roles:** Every agent has an explicit system prompt detailing its strict boundaries and role limitations, preventing cross-pollution of thoughts.
2. **Parallel Orchestration:** Instead of a slow sequential chain, CrewAI allows the Weather, Regulatory, and Renewable agents to run in parallel in Phase 1, followed by the Grid Monitor and Market agents in Phase 2. The entire timeline takes just ~9 seconds, achieving double the intelligence in half the wall-clock execution time.
3. **Structured Tool Inputs:** CrewAI abstracts the Open-Meteo, gridstatus, ChromaDB, and ERCOT database pipelines natively into direct executable tools for the Llama 3 nodes.

---

## 6. Tools & Data

### Tools

| Tool | Built By | Used By | What It Does |
| --- | --- | --- | --- |
| `fetch_texas_weather` | DeMarcus Crump | Weather Analyst | Pulls current temp, wind, conditions for 5 TX cities via Open-Meteo. |
| `search_compliance_database` | Yoana Cook | Reg. Knowledge | RAG queries over NERC/ERCOT regulatory document library. |
| `estimate_renewable_output` | Yoana Cook | Renewable Forecaster | Calculates MW output from solar/wind using Open-Meteo irradiance data. |
| `fetch_ercot_grid_data` | DeMarcus Crump | Grid Monitor | Pulls live demand, total capacity, and fuel mix from ERCOT via GridStatus. |
| `predict_expected_load` | Yoana Cook | Grid Monitor | LSTM model — predicts next-hour load, flags >5% deviation as anomaly. |
| `fetch_realtime_prices` | Yoana Cook | Market Analyst | Grabs live $/MWh settlement prices at the 4 major ERCOT trading hubs. |
| `save_dispatch_report` | DeMarcus Crump | Grid Operator | Pydantic-enforced generation of the final JSON dispatch output. |

### Data Sources

**Open-Meteo API (Free):** Live weather, solar irradiance, and hub-height wind velocities.

**GridStatus.io / ERCOT (Free Python API):** System real-time load, pricing, history.

**Historical ERCOT Database (gridguard.db):** 18,742+ records of ERCOT telemetry with 40 engineered features including stress index, reserve margin, load ramp rates, rolling averages, and lag features.

**NERC/ERCOT Regulatory Document Library:** 25+ manually collected documents including ERCOT SOL Methodology (FAC-011-4), Load Shed Tables (EEA Level 3 thresholds), Wind Integration Reports (2010-2016), and the FERC/NERC Uri 2021 joint post-mortem.

**ChromaDB Local Vector Base:** Embedded versions of the regulatory document library for production RAG retrieval.

---

## 7. Build Plan

| Week | Dates | What Gets Done |
| --- | --- | --- |
| **1** | Apr 3 - Apr 9 | Merge both sets of completed agents into shared repo. Verify all tools connect to the ERCOT data pipeline and SQLite database. |
| **2** | Apr 10 - Apr 16 | Stand up ChromaDB with regulatory PDFs. Test Compliance agent RAG retrieval against real NERC standards. |
| **3** | Apr 17 - Apr 23 | Build and test Fan-Out / Fan-In async orchestration. All 6 agents running in parallel phases end-to-end. |
| **4** | Apr 24 - Apr 30 | Polish structured JSON output. Validate Pydantic schemas against regulatory thresholds from Load Shed Tables. |
| **5** | May 1 - May 3 | Record the 8-minute demo video. Final testing, cleanup, and Final Project submission. |

---

## 8. Expected Challenges

**Challenge 1: Data Overload from API Responses**
Raw JSON responses from APIs like Open-Meteo and ERCOT are massive. Handing them directly to the LLMs causes the agents to lose focus.
*Plan:* Clean and trim the data meticulously within the Python tools before it is handed to the agent. Provide only the 5 or 6 critical numeric vectors that matter.

**Challenge 2: LLM Hallucinations in Urgent Dispatches**
In extreme grid conditions, language models are prone to making up dispatch commands that violate federal law or surpass physical capacity.
*Plan:* Enforce explicit Pydantic schemas on the Grid Operator output, mathematically clamping the recommended dispatch MW amounts to known available bounds from the Load Shed Tables.

**Challenge 3: Multi-Agent Timeout Bottlenecks**
Waiting sequentially for 6 agents executing API calls and generating responses takes far too long for a "real-time" dispatch system.
*Plan:* Strictly enforce the Fan-Out / Fan-In model, forcing Phase 1 and Phase 2 to be executed concurrently, dramatically cutting the total execution time overhead.

**Challenge 4: ERCOT API Blocking Cloud IPs**
ERCOT actively blocks cloud IP addresses via HTTP 403 to prevent scraping.
*Plan:* The production pipeline runs locally and is unaffected. Fallback simulation data is built into the affected agents for cloud-based testing environments.

---

*DeMarcus Crump & Yoana Cook | ITAI 2376 Spring 2026*
