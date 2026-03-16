FROM python:3.13-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git wget gnupg2 ca-certificates build-essential \
    nodejs npm chromium \
    && rm -rf /var/lib/apt/lists/*

# Upgrade Node.js to 22 (needed for Gemini CLI + Claude Code + OpenClaw)
RUN npm install -g n && n 22 && hash -r

# ── Install AI Coding Agents ──

# 1. Gemini CLI (Google — free tier, headless + YOLO mode)
RUN npm install -g @google/gemini-cli

# 2. Claude Code CLI (Anthropic — uses Max subscription)
RUN npm install -g @anthropic-ai/claude-code

# 3. Kimi Code CLI (Moonshot — cheap K2.5 model via OpenRouter)
RUN pip install --no-cache-dir kimi-cli || true

# 4. OpenClaw — multi-agent orchestration framework
RUN npm install -g openclaw || pip install --no-cache-dir openclaw || true

# ── ML Dependencies ──
RUN pip install --no-cache-dir \
    numpy>=2.0 scikit-learn>=1.5 xgboost>=3.0 lightgbm>=4.0 \
    catboost>=1.2 optuna>=4.0 nba_api>=1.4 gradio>=5.0 \
    geopy>=2.4 requests flask gunicorn pyyaml

# ── OpenRouter free model env vars ──
ENV OPENROUTER_MODEL_HEALER=openrouter/healer-alpha
ENV OPENROUTER_MODEL_HUNTER=openrouter/hunter-alpha

# ── App setup ──
WORKDIR /app
COPY . /app/

# Create data dirs + OpenClaw config dir
RUN mkdir -p /app/data/historical /app/data/models /app/data/results /app/logs /app/.openclaw

# Expose Gradio port
EXPOSE 7860

# Launch orchestrator
CMD ["python", "orchestrator.py"]
