# Use an official Python base image (matching the Conda setup)
FROM python:3.12-slim

WARNING: This container must be run with the .env file containing the Groq API keys.

WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core GridGuard multi-agent codebase
COPY . .

# Expose the port for the Flask Dashboard
EXPOSE 5001

# Start the Flask Orchestrator
CMD ["python", "main.py"]
