# Laya vs. Microsoft-Decision-1 vs. CLM

**Research date:** 2026-10-14  
**Scope:** `convaiinnovations/laya`, `Microsoft-Decision-1`, and the CLM implementation in this repository.

## Executive summary

All three systems are **decision models**, not chat models. They take a state
(text or JSON) plus bounded questions and return numerical answers. Their APIs
share the same three core question types:

- `noul`: probability that a statement is true;
- `choice`: one selected option plus a distribution over options;
- `score`: a probability-weighted value over an ordered rubric.

That similarity makes migration and side-by-side evaluation much easier than a
comparison with a generative LLM. The important differences are model scale,
hosting, multilingual coverage, customization, operational burden, and how
strongly each performance claim has been validated.

| Decision factor | Laya | Microsoft-Decision-1 | CLM in this repository |
|---|---|---|---|
| Best fit | Open-weight, local, multilingual decision workloads | Managed, high-volume production decisions in Azure | Research, custom projection-head training, and local controlled deployments |
| Core architecture | 421M ModernBERT-large English model; 322M mmBERT multilingual variant | Qwen3.5-9B post-trained for single-pass decisions | Frozen Qwen3-8B embeddings plus separate state/action projection heads |
| Hosting | Self-hosted Python, HTTP, ONNX, CPU/GPU | Microsoft Foundry managed deployment | Self-hosted Qwen embedding service plus CLM FastAPI server |
| Weights/license | Open weights, Apache-2.0 | Managed model; use governed by Microsoft/Azure product terms | Source and released head are Apache-2.0; Qwen model terms also apply |
| Languages | Router supports 100+ languages; model card reports useful performance in 45/51 tested languages | Microsoft reports multilingual benchmarks, but the public material reviewed does not state a language count | No multilingual routing or repository benchmark; inherits some encoder capability but is not established as multilingual |
| Stated latency | 32.8 ms for multilingual or 39.5 ms for English, one question, Tesla T4, warm | Microsoft reports about 35x faster than GPT-6 Sol and 2.5x faster than H2O-Lightning-4B v1.1; no absolute p50 in the reviewed public text | Deployment-dependent; no canonical end-to-end latency claim in this repository |
| Context | English 512; multilingual 1,024 by default and up to 8,192 | Not specified in the reviewed public model page/docs | Local setup defaults to a 2,048-token Qwen server limit; actual accepted context depends on embedding-server configuration |
| Customization | Documented domain fine-tuning and calibration workflow | No customer fine-tuning workflow documented in the reviewed material | Full projection-head fine-tuning code, hard/soft targets, InfoNCE or soft cross-entropy |
| Price model | No model API fee when self-hosted; infrastructure still costs money | $0.042 per million input tokens; output tokens free (published launch price) | No model API fee when self-hosted; GPU, storage, and operations costs remain |
| Operational burden | Low-to-medium; one Python model/router, with optional server | Low model-operations burden; Azure resource, deployment, identity, quota, and networking remain | Highest; Qwen3-8B pooling server, checkpoint service, GPU capacity, and cache management |

### Recommendation

1. **Choose Microsoft-Decision-1** when a managed Azure service, enterprise
   identity, global/data-zone deployment, very low usage price, and production
   scale matter more than weight access or custom training.
2. **Choose Laya** when open weights, on-premises operation, CPU/ONNX options,
   multilingual routing, or direct domain fine-tuning are requirements.
3. **Choose CLM** when the goal is to experiment with contrastive state/action
   representations, own the complete training loop, or integrate a custom
   projection head with an existing Qwen3-8B embedding stack. It currently
   requires more infrastructure and has less public evidence for general
   zero-shot decision quality than the other two.
4. Do not select from vendor/model-card benchmarks alone. Run the accompanying
   notebook on representative, labeled cases and compare accuracy, calibration,
   latency distributions, abstention thresholds, and total operating cost.

## 1. Common decision contract

The three systems are close enough at the request level to use one evaluation
dataset.

```json
{
  "state": {
    "message": "Checkout is down after a production deployment.",
    "rollback_available": true
  },
  "questions": {
    "owner": {
      "type": "choice",
      "instructions": "Which team should lead the response?",
      "criteria": {
        "application": "Application and deployment owners",
        "database": "Database reliability engineers",
        "network": "Network and edge engineers"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "Assign incident severity.",
      "criteria": ["Minor", "Limited", "Major", "Critical"]
    },
    "customer_impacting": {
      "type": "noul",
      "instructions": "Is this actively affecting customers?"
    }
  }
}
```

The shared contract has practical benefits:

- schemas are explicit and machine-validated;
- there is no generated prose to parse;
- multiple related questions can share a single state;
- probability thresholds can drive automation or human escalation;
- the same labeled examples can evaluate all three backends.

The output probabilities should not be treated as universally calibrated.
Calibration must be tested on the deployment's own traffic, after prompt/schema
design and any fine-tuning.

## 2. Architecture and inference

### Laya

The root `convaiinnovations/laya` checkpoint is a 421M-parameter model:

- fully fine-tuned ModernBERT-large backbone;
- a decision head with two transformer layers;
- one option-marker scorer and an act/escalate head;
- request-time option definitions, so a new label schema does not require
  changing the output vocabulary;
- all questions in a call handled in one forward pass.

The family also contains a 322M mmBERT multilingual checkpoint and a
typed-decisions checkpoint. The recommended `Router` detects language/script
and sends input to the English or multilingual checkpoint. The model card says
the checkpoints were trained using RLCD: reinforcement learning where strictly
proper scoring rules reward honest probability distributions.

This design is compact and operationally straightforward. It also means the
head has a finite token budget shared across options. The model card explicitly
reports weak default results on very high-cardinality classification
(Banking77) and recommends validation/tuning for such workloads.

### Microsoft-Decision-1

Microsoft states that Decision-1 post-trains Qwen3.5-9B for fast, single-pass
decision scoring. It supports the same bounded yes/no, choice, rating, and
rubric-grading pattern. It returns typed numerical results and no rationale.

The model is exposed through a Microsoft Foundry provider endpoint:

```text
<resource-endpoint>/providers/microsoft/v1/systemone
```

The request's `model` value is the deployment name. Version 1 supports
`GlobalStandard` and, in selected regions, `DataZoneStandard`. GlobalStandard
offers global capacity but inference can occur in any supported region;
DataZoneStandard keeps inference in the selected data zone. Neither deployment
type by itself guarantees a latency target.

The managed service removes weight serving and accelerator management, but the
application still owns authentication, retries, observability, thresholds,
human-review paths, and regional/compliance design.

### CLM

CLM is a two-stage contrastive scorer:

1. a Qwen3-8B pooling service produces 4,096-dimensional embeddings;
2. separate MLP state and action heads project embeddings, normally to 512
   dimensions;
3. projected vectors are L2-normalized;
4. each candidate is scored with scaled cosine similarity;
5. softmax across the candidates produces the answer distribution.

For every question, the state representation contains the state followed by
the question instructions. Each candidate is embedded from its option
description. `clm-latest` uses trained projection heads; `clm-raw` is an
ablation that scores in the encoder's original embedding space.

The architecture is especially useful for candidate ranking and for caching
repeated action/option embeddings. It is not a single compact model process:
the reference deployment needs a Qwen3-8B embedding server and a CLM API
process. The Windows documentation recommends WSL2/Linux and an NVIDIA GPU with
about 16 GB or more VRAM for real inference.

## 3. Capability comparison

### Classification and routing

All three are natural fits. Microsoft-Decision-1 has the strongest published
vendor claim: best accuracy over a 36-benchmark comparison covering nearly
150,000 blind questions. Laya has an open implementation and detailed public
model-card results, but its base English checkpoint scored below the
majority-class baseline on Laya's typed-decisions benchmark; the specialized
fine-tuned checkpoint improved from 0.362 to 0.766 accuracy. CLM exposes the
same primitive and can fine-tune against a label distribution, but this
repository does not publish an equally broad decision benchmark.

### Ranking and recommendations

CLM's state/action contrastive objective is directly aligned with ranking a
small candidate set and includes a dedicated `/v1/rank` endpoint. Laya and
Microsoft-Decision-1 can express ranking as a choice or an ordered score.
For large candidate sets, measure both quality and truncation effects. Laya's
model card warns that more than about 20 options can degrade without increasing
the decision-head budget or tuning.

### Binary guardrails

The `noul` primitive makes all three suitable as one signal in a guardrail
pipeline. Microsoft explicitly lists agent actions, prompt injection, content
safety, and production-data access as examples. Laya lists moderation and
guardrails. CLM can express the same gates but does not ship a claim that its
reference head is a certified safety classifier.

No backend should be the only control for high-impact actions. Use deterministic
authorization, policy enforcement, sandboxing, and human approval where
appropriate.

### Ordinal scoring and grading

All three return a probability distribution over ordered levels and a
probability-weighted score. This is valuable for severity, priority,
helpfulness, and quality rubrics. The numerical value is relative to the
provided rubric; it is not an interval-scale measurement. Validate thresholds
against labeled examples and monitor drift.

### Multilingual input

Laya provides the clearest multilingual story: automatic language/script
routing, a 100+ language checkpoint, and published cross-language tests. The
model card also documents an important failure mode: the English checkpoint
can remain highly confident while being wrong on non-Latin scripts, so use the
Router rather than confidence gating.

Microsoft reports multilingual benchmarking but the reviewed catalog and Learn
pages do not specify supported languages or per-language results. CLM uses a
Qwen encoder with multilingual knowledge, but the projection head and current
documentation do not establish multilingual quality. Treat both as
benchmark-required for every target language.

## 4. Deployment and operations

| Area | Laya | Microsoft-Decision-1 | CLM |
|---|---|---|---|
| Quick start | `pip install laya`, then `Router.predict` | Deploy from Foundry catalog, then call provider endpoint | Start Qwen pooling service, download head, start `clm-serve` |
| Authentication | None in-process; optional API key for server | Microsoft Entra ID recommended; API key supported | Optional bearer API key |
| Warm-up | Checkpoint construction can take seconds; preload recommended | Managed by service | Qwen and head startup/download; embedding server must be healthy |
| Scaling | Application-managed replicas/batching | Azure-managed model deployment capacity | Application-managed embedding and API replicas |
| Offline/on-premises | Yes | No managed offline option documented | Yes |
| Observability | Application/server instrumentation | Azure/Foundry ecosystem plus application telemetry | Application/FastAPI instrumentation |
| Failure isolation | Single model/router can be embedded or served | Remote service dependency | Two-service dependency unless embedded differently |

## 5. Cost comparison

- **Laya:** open-weight Apache-2.0 model with no mandatory per-token fee.
  Hardware, hosting, engineering, monitoring, and idle accelerator capacity are
  still real costs.
- **Microsoft-Decision-1:** Microsoft published a launch price of **$0.042 per
  million input tokens** with free output tokens. Confirm current regional
  pricing before procurement.
- **CLM:** Apache-2.0 repository and head, with no mandatory inference fee.
  A continuously running Qwen3-8B GPU service is likely the dominant direct
  cost. The API's `billing_units` field counts questions; it is not a public
  cloud invoice.

At low or bursty volume, Decision-1's managed usage pricing can be much cheaper
than an always-on GPU. At sustained high utilization or where existing GPU
capacity is already available, Laya or CLM may be economically attractive.
Use measured throughput and duty cycle rather than list price alone.

## 6. Training, tuning, and ownership

### Laya

- open weights;
- documented fine-tuning notebook;
- domain temperature fitting for calibration;
- specialized typed-decisions checkpoint;
- easiest of the three for teams wanting a compact, modifiable end-to-end
  decision model.

### Microsoft-Decision-1

- managed inference and no reviewed customer fine-tuning path;
- schema and instructions are the primary customization surface;
- strongest fit when operational simplicity and the managed Azure boundary are
  more important than weight-level control.

### CLM

- frozen encoder with trainable state/action projection heads;
- supports choice-task hard or soft targets;
- supports bidirectional in-batch InfoNCE and per-question soft cross-entropy;
- checkpoints are small compared with the encoder and can be hot-reloaded;
- best experimentation surface, but teams own the data pipeline, training
  evaluation, calibration, deployment, and regression testing.

## 7. Safety and governance

Microsoft's model card says Decision-1 is not designed to be the sole automated
decision-maker for consequential decisions about people, including credit,
employment, housing, insurance, education, healthcare, or legal rights. It
also excludes surveillance/profiling and unlawful use. The model is text-only
and provides no rationale.

Apply the same engineering standard to all three systems:

1. define bounded options and criteria clearly;
2. keep deterministic policy outside the model;
3. choose thresholds on representative labeled data;
4. add an abstain/escalate path;
5. log model/version, schema, probabilities, latency, and final action;
6. evaluate subgroups and languages separately;
7. prevent sensitive data leakage to remote services;
8. require human review for consequential or irreversible outcomes.

## 8. Benchmarking guidance

The published numbers are not directly comparable:

- different datasets, prompts, hardware, network boundaries, warm-up state,
  batch sizes, and model variants were used;
- Laya's latency is a self-reported warm local T4 measurement;
- Microsoft's comparative latency/accuracy is a vendor benchmark and the
  reviewed launch text does not provide the absolute Decision-1 p50;
- CLM has no equivalent broad benchmark in this repository.

For a production decision, collect at least:

- exact-match accuracy or macro-F1 for `choice`;
- Brier score, log loss, and expected calibration error;
- ordinal MAE for `score`;
- threshold precision/recall and coverage for automation;
- p50/p95/p99 cold and warm latency;
- throughput at expected concurrency;
- error and timeout rates;
- cost per 1,000 business decisions;
- accuracy after paraphrasing, option reordering, and harmless formatting;
- per-language and relevant subgroup metrics.

The companion notebook,
`CLMClient/notebooks/decision_models_comparison.ipynb`, provides a shared
three-use-case harness. Its sample labels are illustrative, not a sufficient
benchmark.

## 9. Decision matrix

Score each criterion from 1 (weak) to 5 (strong) and adjust weights for the
actual workload.

| Criterion | Suggested weight | Laya | Microsoft-Decision-1 | CLM |
|---|---:|---:|---:|---:|
| Managed operations | 15% | 2 | 5 | 1 |
| Open weights / on-premises | 15% | 5 | 1 | 5 |
| Multilingual readiness | 10% | 5 | 3 (benchmark required) | 2 (benchmark required) |
| Domain fine-tuning | 15% | 5 | 1 | 5 |
| Low-volume cost | 10% | 3 | 5 | 2 |
| Existing Azure governance | 10% | 2 | 5 | 2 |
| Research flexibility | 10% | 4 | 1 | 5 |
| Simple local footprint | 5% | 4 | 5 (remote API) | 1 |
| Public benchmark evidence | 10% | 4 | 4 | 2 |

These scores are architectural judgments, not measured results. Replace them
with notebook results and organizational constraints.

## Sources

External sources were accessed on 2026-10-14.

1. [Laya model card](https://huggingface.co/convaiinnovations/laya)
2. [Laya Hugging Face model metadata](https://huggingface.co/api/models/convaiinnovations/laya)
3. [Laya documentation](https://nandhakishorm.github.io/laya/)
4. [Microsoft-Decision-1 catalog page](https://ai.azure.com/catalog/models/Microsoft-Decision-1)
5. [Deploy and use Microsoft-Decision-1 in Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision)
6. [Microsoft-Decision-1 launch and benchmark summary](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)
7. Repository implementation: `src/clm/engine.py`, `src/clm/heads.py`,
   `src/clm/schema.py`, `src/clm/client.py`, `train/finetune.py`
8. Repository deployment guide: `CLMClient/SETUP_GUIDE.md`

### Evidence cautions

- Laya and Microsoft benchmark claims are reported by their respective model
  publishers unless otherwise noted.
- Model cards, service prices, regions, and versions can change.
- "Calibrated" describes a training/evaluation goal, not a guarantee on unseen
  production traffic.
- Absence of a documented capability is reported as "not specified" rather
  than treated as proof that the capability does not exist.
