# TalginNosana — Nosana checkpoint

Updated: 2026-09-27

This document is the handoff checkpoint for continuing the Nosana / Decentralize AI work after the $70 GPU credit claim is reviewed.

## Current status

- Public repository: https://github.com/talgat1518/TalginNosana
- The $70 Nosana GPU credit claim has been submitted.
- Project name used for the claim: **TalginNosana — Decentralized Digital Agronomist Runtime**.
- Team member GitHub handle: **@talgat1518**.
- The application states that the project will be deployed/tested on Nosana.
- Social link was not included in the submitted claim. This is optional; do not submit a duplicate claim only to add it.

## Public runtime already built

The repository includes:

- TalginAI V4 public architecture;
- 66,207,360 parameter configuration;
- decoder-only Transformer;
- 12 layers, width 640;
- 10 query heads / 2 KV heads;
- SwiGLU FFN 1792;
- custom 20,480-token vocabulary;
- RoPE, GQA, RMSNorm and tied embeddings;
- FastAPI runtime with /health, /model-info, /generate and /agent/ask;
- verified-context boundary for the Digital Agronomist;
- synthetic farm / weather / satellite demo;
- SHA-256 run evidence for model/context/prompt/output;
- benchmark tooling;
- Dockerfile;
- Nosana job definition;
- tests and public documentation;
- helper scripts for mounting private TalginAI model artifacts locally.

## CI / container status

GitHub Actions currently passes successfully.

Successful workflow commit:
- fb8b2f46ec6aba2dbd0fac452224a86d0c604c37

Published container:
- ghcr.io/talgat1518/talginnosana:latest

Container image digest:
- sha256:42ca028ade34cf0173da1f2b1ef957bc937c3460973f2d484cfe0be6562db450

Before the first Nosana deployment, confirm the GHCR package is publicly pullable.

## Project boundary

TalginAI and TalginAgent are two connected but separate branches:

1. **TalginAI** — own from-scratch language model. Current V4 is experimental and is not claimed to be a production agronomy model.
2. **TalginAgent** — Digital Agronomist application with farm/field/season/crop/weather/tasks/observations/equipment workflows and verified tools.

TalginNosana is the public decentralized compute layer connecting those branches.

Target architecture:

local/private farm data
→ verified local tools
→ allow-listed verified context
→ TalginNosana
→ Nosana decentralized GPU
→ TalginAI / compatible model
→ response
→ reproducible run manifest

## Nosana documentation guidance to remember

From the Decentralize AI claim page and Nosana documentation:

- Use Nosana Deploy / credits after approval.
- Avoid default long-running/infinite deployment behavior during experiments.
- Prefer **SIMPLE** strategy for first tests.
- Use **1 replica**.
- Start with a short container timeout, approximately **20–30 minutes**.
- Choose the cheapest GPU market that satisfies the model VRAM requirement.
- TalginAI 66M already fits on a 4 GB RTX 3050 Laptop, so expensive top-end GPUs are unnecessary for first inference tests.
- Use the existing /health endpoint for a Nosana health check if supported by the current job schema.
- Validate the final job definition with Nosana's official job-definition validator before spending credits.
- Nosana supports long-running HTTP services exposed from containers.
- Nosana can use external model resources such as supported S3/HuggingFace-style resources; do not automatically bake private production weights into the public GitHub repository.
- Ordinary jobs should be treated as potentially public. Never put credentials, private signed URLs, secrets, private farm data, cadastral acquisition methods or sensitive model artifacts in a public job definition.
- If private artifacts must be sent through the network, investigate Nosana Confidential Jobs or another supported private resource mechanism first.
- Node/container/model caching may make repeated runs faster; record cold-start and warm-run results separately.

## First deployment plan after credit approval

1. Open the approval email and verify how the $70 credits were assigned.
2. Log in to Nosana Deploy and confirm credit balance.
3. Create an API key if needed for API/SDK automation.
4. Verify that ghcr.io/talgat1518/talginnosana:latest is publicly pullable.
5. Re-check the current Nosana job schema/documentation.
6. Update nosana/job.json if required.
7. Add /health-based health checking where supported.
8. Validate the job definition with the official Nosana validator.
9. First run:
   - SIMPLE strategy;
   - 1 replica;
   - 20–30 minute timeout;
   - cheap GPU market with enough VRAM;
   - no private production weights;
   - verify container start, CUDA, GPU info, /health, /model-info and /demo/agronomist.
10. Second run:
   - mount or otherwise provide a compatible TalginAI checkpoint + tokenizer using a safe/private mechanism;
   - verify SHA-256 of checkpoint and tokenizer;
   - run /generate and /agent/ask.
11. Benchmark:
   - GPU model;
   - VRAM;
   - cold-start time;
   - model load time;
   - first-response latency;
   - generation latency;
   - tokens/sec;
   - repeated/warm-run timing;
   - compare against local RTX 3050 Laptop.
12. Demonstrate the complete Digital Agronomist flow:
   - local verified field context;
   - Nosana GPU inference;
   - response;
   - run manifest containing model/context/prompt/output hashes.
13. Preserve non-sensitive evidence for judging: logs, screenshots, benchmark JSON and job/run IDs.
14. Prepare demo video and technical article only after a real successful Nosana run.

## Do not publish

Do not add these to the public TalginNosana repo:

- production TalginAI checkpoints unless explicitly chosen for open release;
- private training corpora;
- real farm databases;
- cadastral collectors / HAR files / acquisition endpoints;
- owner/BIN retrieval logic;
- credentials, API keys or .env;
- private signed storage URLs;
- private runtime histories.

A separate open hackathon checkpoint can be created later if reproducibility for judges is worth exposing that specific checkpoint.

## Main judging story

The project should not be presented as merely "a 66M model" or "an agriculture app".

Core story:

**TalginNosana is a decentralized runtime for digital specialists: private local data + verified tools + replaceable AI providers + decentralized GPU inference + reproducible run evidence. Digital Agronomist is the first concrete use case, and TalginAI 66M is the own-weight model used to prove the architecture.**
