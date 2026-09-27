# TalginNosana

**Private data. Verified tools. Decentralized inference.**

TalginNosana is an open hackathon runtime for decentralized AI workloads on the Nosana GPU network.

The first real-world use case is a **Digital Agronomist**. The production TalginAgent project already manages fields, seasons, crops, works, observations, weather, equipment and verified tool results. TalginNosana extracts the safe, reproducible runtime layer needed to run model inference on decentralized GPUs without publishing private farm data or proprietary collectors.

## Why this project exists

TalginAI is a from-scratch language-model project. The current V4 architecture has **66,207,360 parameters**, a custom **20,480-token tokenizer**, GQA, RoPE, RMSNorm and SwiGLU. The main model weights and training corpora remain private.

TalginNosana demonstrates a different boundary:

1. business data and tool execution stay local;
2. only an explicit verified context is sent to the model runtime;
3. inference can run on a Nosana GPU;
4. the response returns to the local digital specialist;
5. every run can be recorded as a reproducible manifest.

The model layer is replaceable. TalginAI is one provider; the agent runtime is not hard-wired to a single model.

## Public scope

This repository contains the model architecture/configuration, safe verified-context runtime, an inference API, a Nosana job definition, Docker packaging, a synthetic agronomy demo and benchmark tooling.

It intentionally does **not** contain production TalginAI weights, training corpora, real farm databases, Kazakhstan cadastral collectors, owner/BIN retrieval logic, credentials or private endpoints.

## Quick start

```bash
docker build -t talginnosana .
docker run --rm -p 8000:8000 talginnosana
```

Available endpoints:

- `GET /health`
- `GET /model-info`
- `GET /demo/agronomist`
- `POST /generate`
- `POST /agent/ask`

The container starts without private model artifacts and exposes the deterministic verified-context demo. To enable TalginAI inference, mount a compatible checkpoint and tokenizer:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e TALGIN_CHECKPOINT_PATH=/model/model.pt \
  -e TALGIN_TOKENIZER_PATH=/model/tokenizer.json \
  -v /local/private/model:/model:ro \
  talginnosana
```

## Nosana

`nosana/job.json` defines a GPU-backed service on Nosana. The container image placeholder is `ghcr.io/talgat1518/talginnosana:latest`; the image still needs to be built/published before the first network run.

## Evidence over claims

TalginAI V4 is experimental and is **not** presented as a production-grade agronomy model. The Digital Agronomist application and the own-weight language model are two parallel parts of the wider project. Nosana is used to test the compute boundary between them.

See [Architecture](docs/ARCHITECTURE.md), [Model card](docs/MODEL_CARD.md) and [Hackathon plan](docs/HACKATHON.md).
