# GridGuard-AI Multi-Agent Orchestrator
# Builds a self-contained Flask SSE dashboard image. The container ships
# the trained ML/DL artifacts (models/) and the historical SQLite DB
# (data/gridguard.db) so it runs offline against scenario replays.
#
# Required at runtime:
#   - GROQ_API_KEY  (Llama-3.3-70B inference; Compliance/Weather/Market/Operator agents)
# Pass via:
#   docker run --env-file .env -p 5001:5001 gridguard-ai
# or:
#   docker run -e GROQ_API_KEY=... -p 5001:5001 gridguard-ai

FROM python:3.12-slim

# System deps for tensorflow / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer caches across code edits
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (.dockerignore-style exclusions live in .gitignore)
COPY . .

# Flask SSE dashboard
EXPOSE 5001

CMD ["python", "main.py"]
