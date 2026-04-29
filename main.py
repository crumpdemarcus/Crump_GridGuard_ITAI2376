import os
import sys
import re
import queue
import threading
import requests
import time
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request

# Resolve paths relative to this file so the server can be launched from any cwd
_REPO_ROOT = Path(__file__).resolve().parent
_LSTM_PATH = _REPO_ROOT / "models" / "load_forecaster.h5"
_RF_PATH = _REPO_ROOT / "models" / "random_forest.pkl"
_CHROMA_PATH = _REPO_ROOT / "data" / "chroma_db"
_DB_PATH = _REPO_ROOT / "data" / "gridguard.db"

# Import deferred to avoid Python 3.9 startup crashes

app = Flask(__name__)

# Queue to hold stdout streams
log_queue = queue.Queue()

# Custom stdout interceptor
class StreamToQueue:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, text):
        if text.strip():  # ignore empty returns
            # Send to terminal
            self.original_stdout.write(text)
            self.original_stdout.flush()
            # Send to web UI
            log_queue.put(text)

    def flush(self):
        self.original_stdout.flush()

    def isatty(self):
        # Rich/CrewAI checks this to enable color codes; we're not a TTY
        return False

    def __getattr__(self, name):
        # Forward any other attribute (fileno, closed, encoding, etc.)
        # to the wrapped stdout so libraries like Rich don't crash.
        return getattr(self.original_stdout, name)

def run_orchestration():
    """Runs the CrewAI Orchestrator in a background thread."""
    original_stdout = sys.stdout
    sys.stdout = StreamToQueue(original_stdout)
    
    try:
        from src.orchestrator import gridguard_crew
        log_queue.put("[SYSTEM INITIATED] Connecting to ERCOT operations center...\n")
        log_queue.put("[SYSTEM INITIATED] Fan-Out Phase 1 & 2 Agents deploying...\n")
        log_queue.put("[SYSTEM INITIATED] Waiting for all 5 Fan-Out Agents to report back before Fan-In Dispatch...\n")
        
        # Kick off the multi-agent system
        final_result = gridguard_crew.kickoff()
        
        try:
            if hasattr(gridguard_crew, 'usage_metrics') and gridguard_crew.usage_metrics:
                # Format depends on crewai version, handling dict or object
                usage = gridguard_crew.usage_metrics
                tokens = usage.get('total_tokens', 0) if isinstance(usage, dict) else getattr(usage, 'total_tokens', 0)
                if tokens > 0:
                    log_queue.put(f"[METRICS]{tokens}\n")
        except Exception:
            pass
        
        log_queue.put("\n=======================================================\n")
        log_queue.put("[EXECUTION COMPLETE] FINAL DISPATCH STRATEGY GENERATED:\n")
        log_queue.put("=======================================================\n")
        log_queue.put(str(final_result) + "\n")
        log_queue.put("[END_OF_STREAM]\n")
    except Exception as e:
        log_queue.put(f"[CRITICAL ERROR] Execution failed: {str(e)}\n")
        log_queue.put("[END_OF_STREAM]\n")
    finally:
        # Restore normal stdout
        sys.stdout = original_stdout

@app.route('/')
def index():
    """Renders the transparent UI Dashboard."""
    return render_template('index.html')

@app.route('/start_execution', methods=['POST'])
def start_execution():
    """API endpoint to click the 'Run Agent' button on the frontend.

    Optional JSON body: {"scenario": "LIVE" | "HEATWAVE_2023" | "STORM_URI_2021"}.
    The selected scenario is exported as the GRIDGUARD_SCENARIO env var so
    every tool checks it (see src/scenarios.py).
    """
    # Clear the queue
    while not log_queue.empty():
        log_queue.get_nowait()

    # Pick up the scenario selection from the frontend (defaults to LIVE).
    payload = request.get_json(silent=True) or {}
    scenario = str(payload.get("scenario", "LIVE")).upper()
    from src.scenarios import SCENARIOS
    if scenario not in SCENARIOS:
        scenario = "LIVE"
    os.environ["GRIDGUARD_SCENARIO"] = scenario
    print(f"[Orchestrator] Scenario set to: {scenario}")

    thread = threading.Thread(target=run_orchestration)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "Execution started", "scenario": scenario})

@app.route('/telemetry_status')
def telemetry_status():
    """Pings the data sources live to verify connection status."""
    status = {
        "open_meteo": "red",
        "gridstatus": "red",
        "groq": "red",
        "local_ml": "red",
    }

    # Test Open-Meteo API (Weather)
    try:
        if requests.get(
            "https://api.open-meteo.com/v1/forecast?latitude=31.9686&longitude=-99.9018&current_weather=true",
            timeout=2,
        ).status_code == 200:
            status["open_meteo"] = "green"
    except Exception:
        pass

    # Test GridStatus API Network (ERCOT Load)
    try:
        requests.get("https://api.gridstatus.io", timeout=2)
        status["gridstatus"] = "green"
    except Exception:
        pass

    # Test Groq Inference Network (LLM)
    try:
        requests.get("https://api.groq.com", timeout=2)
        status["groq"] = "green"
    except Exception:
        pass

    # Local ML readiness: green if all three artifacts are present,
    # yellow if some are present, red if none are.
    ml_ready = sum([_LSTM_PATH.exists(), _RF_PATH.exists(), _CHROMA_PATH.exists()])
    if ml_ready == 3:
        status["local_ml"] = "green"
    elif ml_ready > 0:
        status["local_ml"] = "yellow"

    return jsonify(status)

# Regex to strip ANSI escape codes (colors, bold, reset, etc.)
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

# CrewAI boilerplate lines to filter out of the live feed
_BOILERPLATE_PATTERNS = [
    "You ONLY have access to the following tools",
    "should NEVER make up tools",
    "Tool Name:",
    "Tool Arguments:",
    "Tool Description:",
    "IMPORTANT: Use the following format",
    "Thought: you should always think about what to do",
    "Action: the action to take, only one name of",
    "Action Input: the input to the action",
    "Observation: the result of the action",
    "Once all necessary information is gathered",
    "Thought: I now know the final answer",
    "Final Answer: the final answer to the original",
    "just the name, exactly as it",
    "enclosed in curly braces, using",
]

def _is_boilerplate(text: str) -> bool:
    """Return True if the line is CrewAI internal prompt noise."""
    return any(pat in text for pat in _BOILERPLATE_PATTERNS)

@app.route('/stream')
def stream():
    """Server-Sent Events (SSE) endpoint to stream logs live to HTML."""
    def generate():
        skip_block = False
        while True:
            # Wait for text to appear in the queue
            line = log_queue.get()
            if "[END_OF_STREAM]" in line:
                yield "data: [END_OF_STREAM]\n\n"
                break

            # Strip ANSI escape codes so the browser gets clean text
            clean_line = _ANSI_RE.sub('', line)

            # Filter out CrewAI boilerplate blocks
            if _is_boilerplate(clean_line):
                skip_block = True
                continue
            # End of a boilerplate block (next real content line)
            if skip_block and clean_line.strip() and not _is_boilerplate(clean_line):
                # Check if this line is ALSO boilerplate continuation
                if clean_line.strip().startswith('```') or clean_line.strip() == '':
                    continue
                skip_block = False

            if skip_block:
                continue

            # Presentation Pacing Delays:
            if "Crew Execution Started" in clean_line or "Task Started" in clean_line:
                time.sleep(0.5)
            elif "Agent: Chief Energy Trading Strategist" in clean_line:
                log_queue.put("\n-> [SYSTEM] All Fan-Out intelligence collected. Initiating final Fan-In logic...\n")
                time.sleep(1.0)
            elif clean_line.strip():
                time.sleep(0.02) # Fast typing effect for standard text
            
            # Format for HTML
            clean_line = clean_line.replace('\n', '<br>')
            yield f"data: {clean_line}\n\n"
            
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    print("Gov-Grade GridGuard-AI Control Server Starting...")
    app.run(host='0.0.0.0', port=5001, debug=True)
