# TalginAI V4 — public model card

## Architecture

- Decoder-only Transformer
- Parameters: **66,207,360**
- Layers: 12
- Width: 640
- Query heads: 10
- KV heads: 2
- FFN hidden: 1792
- Vocabulary: 20,480
- RoPE
- GQA
- RMSNorm
- SwiGLU
- Tied token embedding / LM head
- Foundation training context: 2,048 tokens
- Configuration maximum: 4,096 tokens

## Origin

The architecture, tokenizer and weights are developed as a from-scratch project. It is not based on Llama, Qwen or another pretrained checkpoint.

## Status

TalginAI V4 is experimental. It is **not** claimed to be a production assistant or a production agronomy model.

The project separates the digital-specialist runtime and verified tools from the model provider, allowing the applied product to progress while the own-weight model improves.

## Private artifacts

The production checkpoint, training corpora and private evaluation material are not committed here. A compatible owner-supplied checkpoint/tokenizer can be mounted at runtime. A separate public hackathon checkpoint may be released later.
