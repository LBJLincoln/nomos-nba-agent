#!/usr/bin/env python3
"""
Automatic API Key Rotator — Round-robin + rate-limit failover
=============================================================
Uses ALL available keys across ALL providers. When one hits rate limit,
automatically switches to the next. No wasted capacity.

Providers:
  - OpenRouter (7 keys)
  - Groq (5 keys)
  - Kimi/Moonshot (1 key)
  - OpenAI (1 key)
  - Google/Gemini (1 key)
  - xAI/Grok (1 key)
  - LiteLLM proxy (1 key, routes to all above)
"""

import os, json, time, threading
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIKey:
    provider: str
    key: str
    name: str
    base_url: str
    models: list = field(default_factory=list)
    rate_limited_until: float = 0.0
    total_calls: int = 0
    total_errors: int = 0
    last_used: float = 0.0


class KeyRotator:
    """
    Automatic multi-provider API key rotation with rate-limit failover.

    Usage:
        rotator = KeyRotator.from_env()
        key = rotator.get_key("openrouter")  # returns best available key
        rotator.report_success("openrouter", key)
        rotator.report_rate_limit("openrouter", key, cooldown=60)
    """

    def __init__(self):
        self.keys: dict[str, list[APIKey]] = defaultdict(list)
        self._indices: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self.stats = {
            "total_calls": 0,
            "total_rotations": 0,
            "total_rate_limits": 0,
            "providers": {},
        }

    @classmethod
    def from_env(cls):
        """Load ALL API keys from environment variables."""
        rotator = cls()

        # OpenRouter (7 keys)
        or_keys = [
            ("OPENROUTER_API_KEY", "main"),
            ("OPENROUTER_KEY_STANDARD", "standard"),
            ("OPENROUTER_KEY_GRAPH", "graph"),
            ("OPENROUTER_KEY_QUANTITATIVE", "quant"),
            ("OPENROUTER_KEY_ORCHESTRATOR", "orch"),
            ("OPENROUTER_KEY_PME", "pme"),
            ("OPENROUTER_KEY_SPARE", "spare"),
        ]
        for env_var, name in or_keys:
            key = os.environ.get(env_var)
            if key:
                rotator.add_key(APIKey(
                    provider="openrouter",
                    key=key,
                    name=f"openrouter-{name}",
                    base_url="https://openrouter.ai/api/v1",
                    models=["healer-alpha", "hunter-alpha", "meta-llama/llama-3.3-70b-instruct",
                            "qwen/qwen-2.5-72b-instruct", "google/gemini-2.0-flash-exp"],
                ))

        # Groq (5 keys)
        groq_keys = [
            ("GROQ_API_KEY", "1"),
            ("GROQ_API_KEY_2", "2"),
            ("GROQ_API_KEY_3", "3"),
            ("GROQ_API_KEY_4", "4"),
            ("GROQ_API_KEY_5", "5"),
        ]
        for env_var, name in groq_keys:
            key = os.environ.get(env_var)
            if key:
                rotator.add_key(APIKey(
                    provider="groq",
                    key=key,
                    name=f"groq-{name}",
                    base_url="https://api.groq.com/openai/v1",
                    models=["llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct",
                            "meta-llama/llama-4-scout-17b-16e-instruct",
                            "openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.1-8b-instant"],
                ))

        # Kimi/Moonshot
        kimi = os.environ.get("KIMI_API_KEY")
        if kimi:
            rotator.add_key(APIKey(
                provider="kimi",
                key=kimi,
                name="kimi-main",
                base_url="https://api.moonshot.cn/v1",
                models=["moonshot-v1-8k", "moonshot-v1-32k"],
            ))

        # OpenAI
        oai = os.environ.get("OPENAI_API_KEY")
        if oai:
            rotator.add_key(APIKey(
                provider="openai",
                key=oai,
                name="openai-main",
                base_url="https://api.openai.com/v1",
                models=["gpt-4o", "gpt-4o-mini"],
            ))

        # Google/Gemini
        google = os.environ.get("GOOGLE_API_KEY")
        if google:
            rotator.add_key(APIKey(
                provider="google",
                key=google,
                name="google-main",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                models=["gemini-2.0-flash", "gemini-1.5-pro"],
            ))

        # xAI/Grok
        xai = os.environ.get("XAI_API_KEY")
        if xai:
            rotator.add_key(APIKey(
                provider="xai",
                key=xai,
                name="xai-main",
                base_url="https://api.x.ai/v1",
                models=["grok-2", "grok-2-mini"],
            ))

        # LiteLLM proxy (master fallback)
        litellm = os.environ.get("LITELLM_MASTER_KEY", os.environ.get("LITELLM_KEY"))
        if litellm:
            rotator.add_key(APIKey(
                provider="litellm",
                key=litellm,
                name="litellm-s7",
                base_url=os.environ.get("LITELLM_PROXY_URL",
                                        "https://lbjlincoln-nomos-rag-engine-7.hf.space") + "/v1",
                models=["smart", "fast", "default"],
            ))

        return rotator

    def add_key(self, key: APIKey):
        self.keys[key.provider].append(key)
        if key.provider not in self.stats["providers"]:
            self.stats["providers"][key.provider] = {"keys": 0, "calls": 0, "rate_limits": 0}
        self.stats["providers"][key.provider]["keys"] += 1

    def get_key(self, provider: str, model: Optional[str] = None) -> Optional[APIKey]:
        """Get next available key for provider (round-robin with rate-limit skip)."""
        with self._lock:
            keys = self.keys.get(provider, [])
            if not keys:
                return None

            now = time.time()
            n = len(keys)
            start_idx = self._indices[provider]

            # Try round-robin, skipping rate-limited keys
            for offset in range(n):
                idx = (start_idx + offset) % n
                key = keys[idx]
                if key.rate_limited_until <= now:
                    if model and model not in key.models and key.models:
                        continue  # Skip if model not supported
                    self._indices[provider] = (idx + 1) % n
                    key.last_used = now
                    key.total_calls += 1
                    self.stats["total_calls"] += 1
                    self.stats["providers"][provider]["calls"] += 1
                    return key

            # All keys rate-limited — return the one that expires soonest
            soonest = min(keys, key=lambda k: k.rate_limited_until)
            wait = soonest.rate_limited_until - now
            if wait > 0 and wait < 120:  # Wait up to 2 min
                time.sleep(wait + 1)
            soonest.last_used = now
            soonest.total_calls += 1
            self.stats["total_calls"] += 1
            self.stats["total_rotations"] += 1
            return soonest

    def get_any_key(self, preferred_providers=None) -> Optional[APIKey]:
        """Get a key from ANY provider, trying preferred ones first."""
        providers = preferred_providers or ["groq", "openrouter", "litellm", "kimi", "openai", "xai", "google"]
        for provider in providers:
            key = self.get_key(provider)
            if key:
                return key
        return None

    def report_success(self, provider: str, key: APIKey):
        """Report successful API call."""
        pass  # Already tracked in get_key

    def report_rate_limit(self, provider: str, key: APIKey, cooldown: int = 60):
        """Mark key as rate-limited for cooldown seconds."""
        with self._lock:
            key.rate_limited_until = time.time() + cooldown
            key.total_errors += 1
            self.stats["total_rate_limits"] += 1
            self.stats["providers"].get(provider, {})["rate_limits"] = \
                self.stats["providers"].get(provider, {}).get("rate_limits", 0) + 1

    def report_error(self, provider: str, key: APIKey, error_code: int = 500):
        """Handle non-rate-limit errors."""
        if error_code == 429:
            self.report_rate_limit(provider, key, cooldown=60)
        elif error_code >= 500:
            self.report_rate_limit(provider, key, cooldown=30)

    def get_status(self) -> dict:
        """Get status of all keys."""
        now = time.time()
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_keys": sum(len(v) for v in self.keys.values()),
            "total_calls": self.stats["total_calls"],
            "total_rate_limits": self.stats["total_rate_limits"],
            "providers": {},
        }
        for provider, keys in self.keys.items():
            available = sum(1 for k in keys if k.rate_limited_until <= now)
            status["providers"][provider] = {
                "keys_total": len(keys),
                "keys_available": available,
                "keys_rate_limited": len(keys) - available,
                "total_calls": sum(k.total_calls for k in keys),
                "total_errors": sum(k.total_errors for k in keys),
            }
        return status

    def summary(self) -> str:
        """Human-readable summary."""
        s = self.get_status()
        lines = [f"KeyRotator: {s['total_keys']} keys, {s['total_calls']} calls, {s['total_rate_limits']} rate limits"]
        for p, info in s["providers"].items():
            lines.append(f"  {p}: {info['keys_available']}/{info['keys_total']} available, {info['total_calls']} calls")
        return "\n".join(lines)


def call_llm(rotator: KeyRotator, system_prompt: str, user_prompt: str,
             provider: str = "openrouter", model: str = "healer-alpha",
             max_tokens: int = 4000, temperature: float = 0.3,
             max_retries: int = 3) -> str:
    """
    Call any LLM with automatic key rotation and failover.
    Tries the preferred provider first, then falls back to others.
    """
    import urllib.request

    providers_to_try = [provider]
    # Add fallback providers — Groq FIRST (free, working, has Kimi K2)
    all_providers = ["groq", "openrouter", "litellm", "kimi", "openai", "xai"]
    for p in all_providers:
        if p not in providers_to_try:
            providers_to_try.append(p)

    last_error = None
    for attempt_provider in providers_to_try:
        for attempt in range(max_retries):
            key = rotator.get_key(attempt_provider)
            if not key:
                break

            # Resolve model for this provider
            actual_model = model
            if attempt_provider == "groq":
                # Use Kimi K2 on Groq for smart tasks, Llama for fast
                if model in ("healer-alpha", "hunter-alpha", "smart"):
                    actual_model = "moonshotai/kimi-k2-instruct"
                else:
                    actual_model = "llama-3.3-70b-versatile"
            elif attempt_provider == "kimi":
                actual_model = "moonshot-v1-8k"
            elif attempt_provider == "openai":
                actual_model = "gpt-4o-mini"
            elif attempt_provider == "xai":
                actual_model = "grok-2-mini"
            elif attempt_provider == "litellm":
                actual_model = "smart"

            body = json.dumps({
                "model": actual_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }).encode()

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key.key}",
            }
            if attempt_provider == "openrouter":
                headers["HTTP-Referer"] = "https://nomos-nba-quant.ai"
                headers["X-Title"] = "NOMOS NBA Quant"

            url = f"{key.base_url}/chat/completions"

            try:
                req = urllib.request.Request(url, data=body, headers=headers)
                resp = urllib.request.urlopen(req, timeout=120)
                data = json.loads(resp.read().decode())
                choices = data.get("choices", [])
                if choices:
                    rotator.report_success(attempt_provider, key)
                    return choices[0].get("message", {}).get("content", "")
            except urllib.error.HTTPError as e:
                rotator.report_error(attempt_provider, key, e.code)
                last_error = f"{attempt_provider}/{key.name}: HTTP {e.code}"
            except Exception as e:
                rotator.report_error(attempt_provider, key, 500)
                last_error = f"{attempt_provider}/{key.name}: {e}"

    return f"[ALL PROVIDERS FAILED] Last error: {last_error}"


# Singleton
_rotator = None

def get_rotator() -> KeyRotator:
    global _rotator
    if _rotator is None:
        _rotator = KeyRotator.from_env()
    return _rotator
