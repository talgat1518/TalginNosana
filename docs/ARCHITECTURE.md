# Architecture

TalginNosana is the public decentralized-compute boundary between a digital specialist and a GPU-backed model.

```text
private farm / business data
          |
          v
local verified tools + database
          |
          v
allow-listed verified context
          |
          v
TalginNosana inference API
          |
          v
Nosana decentralized GPU
          |
          v
TalginAI / compatible model
          |
          v
response + reproducible run manifest
```

## Design principles

### Data sovereignty
Production operational data, cadastral logic, user keys and business history stay outside this public runtime. Only an explicit context object is sent to inference.

### Verified tools before model reasoning
The model does not get arbitrary SQL or arbitrary HTTP access. Production TalginAgent performs deterministic tool calls first, records results, and supplies those facts to the model.

### Provider independence
The digital-specialist layer is separate from the model layer. The same workflow can use local inference or decentralized GPU inference.

### Honest boundary
The public agronomist brief is deterministic and is not called AI. Model-backed generation is enabled only when real model artifacts are mounted.

### Reproducibility
Model-backed responses can include hashes of the verified context, prompt, output, checkpoint and tokenizer. This can later be anchored in permanent storage without exposing private farm data.
