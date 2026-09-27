from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import canonical_hash, deterministic_agronomist_brief, verified_context
from .runtime import TalginRuntime


ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "demo" / "sample_farm.json"

app = FastAPI(title="TalginNosana", version="0.1.0")
runtime = TalginRuntime()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=64, ge=1, le=256)


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    context: dict[str, Any]


@app.get("/health")
def health():
    info = runtime.info()
    return {
        "service": "TalginNosana",
        "status": "ok",
        "model_available": runtime.available,
        "cuda": info["cuda"],
        "gpu": info["gpu"],
    }


@app.get("/model-info")
def model_info():
    return runtime.info()


@app.post("/generate")
def generate(body: GenerateRequest):
    started = time.perf_counter()
    try:
        result = runtime.generate(body.prompt, body.max_new_tokens)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503 if not runtime.available else 422, str(exc)) from exc
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


@app.get("/demo/agronomist")
def agronomist_demo():
    payload = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
    context = verified_context(payload)
    return {
        "mode": "deterministic_verified_context_demo",
        "input": context,
        "brief": deterministic_agronomist_brief(context),
    }


@app.post("/agent/ask")
def agent_ask(body: AgentRequest):
    context = verified_context(body.context)
    brief = deterministic_agronomist_brief(context)
    prompt = (
        "You are a digital agronomist. Use only the verified JSON context below. "
        "Separate facts from hypotheses. Satellite indices show anomalies, not diagnoses. "
        "If evidence is insufficient, request a field inspection.\n\n"
        f"VERIFIED_CONTEXT={json.dumps(context, ensure_ascii=False, sort_keys=True)}\n\n"
        f"LOCAL_BRIEF={json.dumps(brief, ensure_ascii=False, sort_keys=True)}\n\n"
        f"USER_REQUEST={body.message}"
    )

    if not runtime.available:
        return {
            "provider": "none",
            "model_available": False,
            "verified_context_sha256": canonical_hash(context),
            "local_brief": brief,
            "next_step": "Mount a compatible TalginAI checkpoint/tokenizer to enable decentralized model inference.",
        }

    started = time.perf_counter()
    generated = runtime.generate(prompt, 128)
    manifest = {
        "service": "TalginNosana",
        "model": runtime.info(),
        "verified_context_sha256": canonical_hash(context),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(generated["text"].encode("utf-8")).hexdigest(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    return {
        "provider": "talginai",
        "answer": generated["text"],
        "local_brief": brief,
        "run_manifest": manifest,
    }
