# Multi-Team Routing on CLM-8B: Design & Experimentation Plan

Oct 20, 2026 · @Prashant Rajoria

## Summary

Build per-team routing as one frozen Qwen3-8B encoder plus one fine-tuned CLM head per team, selected by the request's `model` field. The repo's server already supports this: `clm-serve --ckpt-dir` serves every head in a folder under its own name, hot-reloads changed heads, and shares one vector cache across heads. No LoRA on the backbone is needed, and adding one would break the heads, which are locked to the stock encoder.

**Hardware verdict:** a single RTX 3090 (24 GB) is enough to prototype, run every experiment in this plan, and serve a pilot. The repo's own playground runs on one 24 GB RTX 4090. Production with several teams and uptime targets needs at least two GPUs, or one larger datacenter card. LoRA on the backbone or a much larger encoder would not fit on a 3090.

**Approach in one line:** start each team zero-shot with typed `Choice` questions built from its taxonomy, measure, then fine-tune that team's head only where zero-shot falls short of its target.

Assumptions to confirm: several teams, each with its own taxonomy, every request tagged with exactly one team, and taxonomies that change occasionally rather than per request.

## How CLM-8B works

CLM scores candidate actions against a state; it never generates text. Per the [repo](https://github.com/Contrastive-LM/CLM) and [model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B):

- **Two heads on a frozen encoder.** A state head and an action head, each about 20M parameters, sit on frozen Qwen3-8B last-token-pooled embeddings. The reference checkpoint is a 75 MB file.
- **Training objective.** Bidirectional InfoNCE over matched state-action pairs, extended with hard negatives in mid-training.
- **Inference.** One encoder pass per new text, then a dot product per cached candidate. A softmax over the scores is the answer distribution.
- **Typed questions.** `POST /v1/systemone` takes a state plus questions of type `Noul` (yes/no), `Choice` (pick one option, each embedded from its description) or `Score` (ordered levels). `Choice` maps directly onto taxonomy routing.
- **Two processes.** A vLLM pooling server runs Qwen3-8B on the GPU; `clm-serve` runs the heads and the vector cache.
- **Fine-tuning.** `train/finetune.py` trains only the heads, with two tasks: `clm` (best-of-N verification) and `choice` (typed decisions, via `train/adapters.py`).

Limitations stated by the authors that shape this design:

- **Encoder-locked:** heads only work with Qwen3-8B last-token pooling, so the encoder can never change underneath a trained head.
- **Relative probabilities:** scores are relative to the candidate set given, so there is no built-in "none of these".
- **States over 2048 tokens are truncated** unless both servers are raised together, which needs more GPU memory.

The Enera write-up adds an independent finding: in tri-state tests CLM put under 2% probability on "uncertain", so abstention must be built around it.

## Requirements

The system must route each request into one category of its team's taxonomy, improve per team over time, and say when it is unsure.

| ID | Requirement | Type |
| --- | --- | --- |
| R1 | Each request carries one `team_id`; routing uses only that team's taxonomy and head | Functional |
| R2 | Teams own their taxonomy: add, rename, retire categories without a code change | Functional |
| R3 | Return top-k categories with probabilities and a confidence value | Functional |
| R4 | Abstain below a per-team threshold and hand off to a fallback | Functional |
| R5 | Per-team accuracy is measured, versioned and can be improved independently | Quality |
| R6 | One team's retraining never changes another team's results | Isolation |
| R7 | Every head and taxonomy version can be rolled back | Operations |
| R8 | Latency target per request, set with the teams (e.g. p95 under 100 ms) | Performance |

Open inputs needed from teams: number of teams, categories per team, flat or hierarchical taxonomy, labeled examples available, typical request length in tokens, and requests per second at peak.

## System architecture

All teams share one frozen Qwen3-8B encoder; a team's identity only selects which 75 MB head scores its request.

&#91;embedded content: system architecture · offline training loop and online request path\]

Offline, each team's taxonomy and labels are embedded once and used to fine-tune its head, which is promoted by hot reload. Online, the gateway picks the team's head and builds a `Choice` question from its taxonomy; low-confidence results go to a fallback, and corrections feed the next training round.

## Data pipeline

Each team contributes a versioned taxonomy and labeled examples; the pipeline turns them into cached embeddings once, so every training run after that is cheap.

**Taxonomy spec (per team, in git).** One YAML file per team: category key, a 1–3 sentence description, optional parent for hierarchies, optional example phrases, and a version number. The description is what CLM embeds for a `Choice` option, so it is the first lever on accuracy.

**Labeled examples.** Each row is `team_id, request_text, category_key, timestamp, source`. Sources, cheapest first:

1. Historical tickets or requests that were already routed by people.
2. Corrections logged in production (a human re-routes a request).
3. Synthetic examples from an LLM, only for categories with fewer than \~50 real examples, and always kept out of test sets.

**Hard negatives.** For each example, the 3–10 sibling categories the zero-shot model confuses it with most. These come from the confusion matrix of the first evaluation, matching CLM's own mid-training recipe.

**State formatting.** Send states as prose or `key: value` text, never JSON, since the README says the heads are trained on prose. Fix one template per team (e.g. title, body, project field) and never change it without re-embedding.

**Splits.** Per team: a time-based split (train on older, test on the most recent 15–20%) to expose drift, plus a held-out category split to test new categories added without retraining. Freeze each test set and version it.

**Embedding cache.** Run every state and option text through the vLLM Qwen3-8B pooling server once, using `train/embed_utils.py` so the token recipe matches training, and store them with `preprocessing/hf_embeddings.py`. Re-embed only new or changed text.

## Training design

The per-team "adapter" is a CLM head checkpoint, fine-tuned from the reference head with `train/finetune.py --task choice`. Three training strategies are compared in the experiments; the winner may differ by team size.

| Strategy | How | Best for |
| --- | --- | --- |
| A. Per-team head | Start from `CLM_v0.1-8B.pt`, fine-tune on one team's data only | Teams with thousands of examples |
| B. Shared org head | One head fine-tuned on all teams' data pooled, each row keeping its own team's option set | Many small teams, shared vocabulary |
| C. Shared then per-team | Start from B, then a short per-team fine-tune | Most teams, if B helps |

**Guarding against overfitting on small teams.** The README's scaling result puts the optimal head size at about 310 training tokens per parameter. A team with 5,000 examples of 200 tokens has about 1M tokens, far below what a 20M-parameter head needs from scratch. So small-team runs must start from a pretrained head and use early stopping, a low learning rate, and one of: training only the last layer, or a small low-rank update on the head layers (a design proposal to test, not a repo feature).

**Replay.** The authors found that mixing 40% general data back in during post-training kept general accuracy at 68.5% versus 56.2% without it. Mirror this: mix a slice of general typed-decision data (e.g. `LocalLLaMA/typed-decisions`, used in the README's example) into each team run, and tune the ratio.

**Loss.** InfoNCE over the team's option set, with mined hard negatives from the data pipeline. Optionally weight rare categories so they are not ignored.

**Run protocol.** Follow the repo's [fine-tuning guide](https://github.com/Contrastive-LM/CLM/blob/main/docs/FINETUNING.md) discipline: a fixed train and eval command, the unchanged baseline first, one change per commit, and a `results.tsv` row per run, kept only if the held-out score improves. Never tune on the test split.

**Output.** `heads/<team_id>/<version>.pt` in the repo's checkpoint format (`state_head`, `action_head`, `logit_scale`, `cfg`), plus a metadata file with the taxonomy version, data snapshot, encoder version and eval scores.

## Serving design

Serving reuses `clm-serve` almost unchanged; the new piece is a thin routing gateway in front of it.

**Head registry.** Promoted heads are written to one folder as `<team_id>.pt` and served with `clm-serve --ckpt-dir`, which exposes each file under its stem as a model name. Writing a new file there triggers the server's hot reload, so promoting or rolling back a team is a file swap.

**Gateway (new, small service).** For each request it:

1. Looks up the team's live head name and taxonomy version.
2. Builds one `Choice` question whose `criteria` are that taxonomy's category descriptions. For a hierarchy, it asks level 1 first, then the chosen branch's children.
3. Calls `POST /v1/systemone` with `model` set to the team's head.
4. Applies the team's confidence threshold and returns top-k, or hands off to the fallback.
5. Logs the request, scores, decision and any later correction for the data pipeline.

**Caching.** The server's vector cache is keyed by head and head generation, so all teams share one memory arena and a reloaded head never reuses stale vectors. Category descriptions are embedded once and stay cached. Measured on an RTX 4090, a new state costs about 28 ms server-side and a cache hit under 1 ms.

**Batching.** Every request names one team, so there is no mixed-head batching problem. The encoder pass is shared across teams anyway, because the encoder is the same; only the small head differs.

**Abstention and fallback.** The API returns `confidence` as the top probability minus the mean of the others. Each team gets a threshold tuned on its validation set to hit a target precision. Below it, the request goes to a fallback: a larger LLM classifier, or a human queue. Add an explicit "other / not sure" option only if experiment E9 shows it helps, since CLM rarely picks uncertainty on its own.

**Memory per team.** Each head is about 75 MB. Whether `--ckpt-dir` keeps all heads loaded on the GPU at once must be checked in `src/clm/heads.py`. If it does, 30 teams is about 2.3 GB, which matters on a 24 GB card; heads can also run on the CPU with `--device cpu`, at some latency cost.

## Experimentation design

The central question is how much per-team fine-tuning beats zero-shot CLM with good category descriptions, and at what data size it starts paying off. Run everything first on two pilot teams, one large and one small, then roll out.

**Hypotheses**

- H1: Rewriting category descriptions alone closes a large share of the gap between zero-shot and fine-tuned accuracy.
- H2: Per-team heads beat a shared org head for large teams; a shared head, or shared then per-team, wins for small teams.
- H3: Hard negatives from the confusion matrix lift accuracy most on sibling categories.
- H4: A confidence threshold gives a better precision and coverage trade-off than adding an explicit "other" option.

**Experiments**

| ID | Experiment | Compares | Phase |
| --- | --- | --- | --- |
| E0 | Zero-shot, option keys only | `clm-latest`, criteria = category names | 1 |
| E1 | Zero-shot, rich descriptions | Same, criteria = written descriptions; 2–3 rewrite rounds | 1 |
| E2 | Raw encoder ablation | `clm-raw` (cosine in encoder space) vs `clm-latest` | 1 |
| E3 | LLM classifier baseline | A frontier LLM prompted with the taxonomy: accuracy ceiling, latency, cost | 1 |
| E4 | Per-team head | Strategy A vs E1 | 2 |
| E5 | Shared org head | Strategy B vs A | 2 |
| E6 | Shared then per-team | Strategy C vs A and B | 2 |
| E7 | Hard negatives | Mined negatives on vs off | 2 |
| E8 | Replay ratio | 0%, 20%, 40% general data | 2 |
| E9 | Abstention | Threshold on `confidence` vs "other" option vs temperature scaling | 3 |
| E10 | Learning curve | 100, 500, 2,000, all examples per team | 3 |
| E11 | Taxonomy shape | Flat vs two-level hierarchy | 3 |
| E12 | Taxonomy change | Add held-out categories without retraining, then with | 3 |
| E13 | Quantized encoder (3090 only) | bf16 vs 8-bit encoder: accuracy drift of trained heads | 3, optional |

**Metrics, per team, on the frozen test split**

- Accuracy: top-1, top-3, macro-F1 (so rare categories count), per-category recall.
- Sibling confusion: error rate between categories that share a parent.
- Selective accuracy: precision at 80% and 90% coverage after thresholding, and the coverage needed for the team's target precision.
- Calibration: expected calibration error of the returned probabilities.
- Cost: p50 and p95 latency, GPU-seconds per 1,000 requests, training time per head.

**Statistics.** Report 95% bootstrap confidence intervals per team. Treat a change as real only if its interval clears the baseline's; with small test sets, use McNemar's test on paired predictions. Use 3 random seeds for every training run.

**Decision gates**

1. After phase 1: if E1 meets a team's target, that team ships zero-shot and skips training.
2. After phase 2: pick strategy A, B or C per team size, using the E10 learning curve to set the data threshold for a dedicated head.
3. After phase 3: set each team's abstention threshold and the retraining trigger (e.g. a drop of 3+ points in weekly macro-F1 on corrected traffic).

## Hardware feasibility: RTX 3090 24 GB

Yes, a 3090 can do the training this plan needs, because the expensive part is a one-time embedding pass through a frozen encoder, and the trained part is small. The repo's playground screenshot was captured on one 24 GB RTX 4090, the same memory as a 3090. Throughput and latency figures below marked "est." are my estimates, not measurements; benchmark them in week 1.

**Memory budget on one 3090**

| Item | Size | Source |
| --- | --- | --- |
| Qwen3-8B encoder weights, bf16 | \~16.4 GB | 8.2B params × 2 bytes |
| vLLM working memory at 2048 tokens | \~3–5 GB | est., set by `--gpu-memory-utilization` |
| CLM vector cache (default 2% of device) | \~0.5 GB | README default |
| One team head | \~75 MB | reference checkpoint size |
| Head training (20M params, optimizer, 512-row batches) | under 1 GB | est. |

**What fits and what does not**

| Workload | On one 3090? | Notes |
| --- | --- | --- |
| Serve the encoder at 2048 tokens | Yes | The setup the README ships |
| Embed 10,000 examples of \~200 tokens | Yes, \~10–25 min | est. 1,500–3,000 tokens/s prefill |
| Embed 1M examples | Yes, \~1–2 days | Run once; cache results |
| Train one team head on cached embeddings | Yes, minutes | Only the heads train |
| All of E0–E13 for 2 pilot teams | Yes | Run embedding and serving at different times |
| Serve a pilot, several teams | Yes | est. 35–50 ms per new state vs 28 ms on a 4090 |
| States of 8,192 tokens | Tight | Needs a lower batch size; test it |
| LoRA or full fine-tune of Qwen3-8B | No, and not wanted | Breaks the encoder lock; would need QLoRA or 80 GB cards |
| A 35B-class encoder | No | \~70 GB in bf16; release status of CLM-35B not verified here |

**3090 operating tips**

- Lower vLLM's `--gpu-memory-utilization` to about 0.80 so the heads and vector cache fit beside it, or run heads on the CPU.
- Do not embed training data and serve live traffic on the same card at once; schedule embedding jobs off-peak.
- The 3090 is Ampere: it supports bf16 but has no hardware FP8, so E13 should test 8-bit weight-only quantization, not FP8.

**What you need beyond the 3090**

| Stage | Hardware | Why |
| --- | --- | --- |
| Prototype and experiments | 1× RTX 3090 24 GB | Enough for everything in phases 1–3 |
| Production, small | 2× 24 GB GPUs (3090, 4090, or datacenter L4/A10-class) | Failover, and embedding jobs off the serving card |
| Production, long states or high traffic | 1–2× 48 GB (L40S-class) or 80 GB (A100/H100) | Room for 8K-token states and larger batches |
| Only if you later change the encoder | 80 GB-class GPUs | Retraining encoder and all heads together |

A consumer 3090 has no ECC memory and no datacenter support, which is fine for a pilot but a risk for a production SLA.

## Build plan

A pilot with two teams takes about 10 weeks on one RTX 3090, and production rollout about 4 more. These durations are my estimates for the staffing below, not figures from the repo.

&#91;embedded content: build roadmap · 5 phases, 3 decision gates, about 14 weeks\]

Each gate can stop a team early: a team that passes Gate 1 ships zero-shot and never needs a trained head.

**What it takes**

| Resource | Amount | Used for |
| --- | --- | --- |
| ML engineer | 1, full time | Data pipeline, embedding, training runs, evaluation |
| Backend engineer | 1, about half time | Gateway, head registry, logging, monitoring |
| Team subject experts | A few hours per team per phase | Category descriptions, label review, target setting |
| GPU, phases 0 to 3 | 1x RTX 3090 24 GB | All experiments and the pilot |
| GPU, phase 4 | +1 GPU, 24 GB or larger | Failover and off-card embedding jobs |
| Storage | Tens of GB | Embedding cache, head versions, logs |

**New code to write:** the gateway service, the taxonomy-to-Choice builder, a data adapter from team exports into the repo's choice format, the per-team eval harness with bootstrap intervals, and a promotion script that writes heads into the ckpt-dir folder.

## Risks and open questions

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Encoder version drift (a vLLM or Qwen3 update changes embeddings) | Every trained head silently degrades | Pin encoder weights and vLLM version; re-run a canary eval on every upgrade |
| Overconfident routing, weak abstention | Wrong routes look certain | Per-team threshold (E9), fallback path, monitor corrections |
| Small teams overfit their head | Worse than zero-shot | Gate 2: ship zero-shot or shared head below the E10 data threshold |
| Long requests truncated at 2,048 tokens | Key context lost | Put the most routing-relevant fields first in the state template; test 8K |
| Many heads exceed 3090 memory | OOM or eviction | Check how `--ckpt-dir` loads heads; heads on CPU or a second GPU |
| Taxonomy descriptions written poorly | Low zero-shot accuracy | Description review in E1; teams own and iterate their text |
| Young project: 9 commits on the inference branch | API and checkpoint format may change | Pin a commit; wrap the client behind the gateway |

**Open questions to answer before phase 1**

- [ ] How many teams, and how many categories each?
- [ ] Flat or hierarchical taxonomies?
- [ ] Labeled examples per team today, and how routing corrections are captured?
- [ ] Typical and maximum request length in tokens?
- [ ] Target precision, coverage and p95 latency per team?
- [ ] Read `train/adapters.py` to confirm the exact `choice` data format, and `src/clm/heads.py` for multi-head memory use.

**Sources:** [CLM repo README](https://github.com/Contrastive-LM/CLM), [fine-tuning guide](https://github.com/Contrastive-LM/CLM/blob/main/docs/FINETUNING.md), [CLM-v0.1-8B model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), [Enera write-up](https://www.eneralabs.com/blog/stanford-nvidia-clm-8b-system-one-agent-routing-enterprise-2026/).
