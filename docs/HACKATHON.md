# Decentralize AI Hackathon plan

## Contribution

TalginNosana demonstrates a reusable runtime for digital specialists with:

- local/private operational data;
- verified and allow-listed tool context;
- replaceable model providers;
- decentralized GPU inference on Nosana;
- reproducible hashes for model-backed runs.

The Digital Agronomist is the first concrete use case.

## Existing work behind the demo

The private production projects already contain:

- a 66.2M-parameter from-scratch language model;
- training, evaluation and own-weight inference code;
- a farm/field/season data model;
- work logs, observations, equipment and irrigation modules;
- current and historical weather integration;
- preliminary water-balance calculations;
- cadastral field workflows;
- an AI-provider abstraction with a TalginAI provider;
- safe deterministic tool execution before model inference.

The hackathon repository publishes only the reusable safe subset.

## Milestones

1. Public reproducible runtime.
2. Build and publish the container image.
3. Deploy the service on a Nosana GPU node and capture node/GPU/runtime evidence.
4. Mount a compatible TalginAI checkpoint and tokenizer.
5. Complete an agent round trip: verified context -> Nosana inference -> response -> manifest.
6. Benchmark laptop RTX 3050 versus Nosana GPU.
7. Persist a non-sensitive verifiable run manifest.
8. Publish a demo video and technical article with measured results and limitations.

## Success criteria

A judge should be able to see source architecture, a public container, a valid Nosana job definition, a real Nosana GPU run, measured inference results, an application flow from verified field data to an agent response, and hashes proving which model/context/output were involved.
