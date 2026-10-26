# CLM email routing: slide-by-slide reading companion

This companion explains all four slides in the [experiment presentation](./clm_email_experiment_findings.pptx). Slide 3 includes a detailed explanation of every reported metric and how to interpret the results.

**Sources:** the [experiment plan](./experiment.md), the [evaluation notebook](../notebooks/clm_email_classification_evaluation.ipynb), and its saved evaluation artifacts. Training and inference were not rerun to prepare this explanation; the reported counts and calibration results were checked against the saved predictions.

**Central finding:** on this synthetic six-class email benchmark, training the CLM projection heads while keeping Qwen3-8B frozen raised test accuracy from **26.88% to 97.59%**. This demonstrates useful task adaptation, but not yet reliable production routing on real enterprise emails.

---

## Slide 1: Specialize the router, not the entire language model

### What problem are we trying to solve?

Different teams need different routing decisions. A support team might route requests by product issue, while a finance team might use invoice, reimbursement, and payment categories. These category sets are called **taxonomies**.

The design needs to support:

- A different taxonomy for each team.
- Changes to category definitions over time.
- Independent quality improvements and rollback.
- A fallback when the model cannot safely choose a category.
- Shared infrastructure rather than a complete large model for every team.

The motivation is therefore not simply "classify emails." It is to find a practical way to specialize routing decisions while sharing the expensive language representation.

### What does the proposed architecture do?

```text
Team ID + request text
          |
          v
Shared frozen Qwen3-8B encoder
          |
          v
Selected team's CLM projection-head checkpoint
          |
          v
Scores over that team's candidate categories
          |
          v
Confidence gate: route automatically or use a fallback
```

The components mean:

| Term | Meaning in this system |
|---|---|
| Encoder | The model that converts text into numerical representations called embeddings. |
| Frozen encoder | Its learned weights do not change during this fine-tuning experiment. |
| Embedding | A numerical representation of text that preserves features useful for comparison and classification. |
| State | The input being classified. In the notebook, it contains the email subject and body. |
| Candidate action | A possible decision. Here, each email category is a candidate. |
| Projection heads | Smaller learned mappings applied to state and candidate embeddings before scoring their compatibility. |
| Team head | Shorthand for the team's checkpoint containing the state/action projections and related scoring parameters; not a separate 8B encoder. |
| Typed `Choice` question | A structured request asking CLM to choose one option from a defined candidate set. |
| Fallback | Another decision path, such as a human queue or a stronger classifier, used when automatic routing is unsafe. |

CLM **scores supplied options rather than generating a free-form answer**. The categories must already be defined.

The plan describes a reference head checkpoint of approximately **75 MB**, much smaller than the encoder. A request's `model` field selects the relevant served head.

### Why not fine-tune the whole encoder?

Sharing a frozen encoder makes the expensive representation reusable. Team-specific changes are concentrated in smaller checkpoints, which are easier to train, version, and deploy independently.

The encoder is also part of the heads' compatibility contract. Changing its weights or embedding recipe can invalidate the learned projections. The reported experiment does **not** use backbone LoRA or full-model fine-tuning.

### Why start with zero-shot evaluation?

**Zero-shot** means using the released model for this taxonomy without training it on the email dataset's labeled training examples. It does not mean the released model has never been trained on anything.

The plan's decision rule is:

> Try the released model with useful category descriptions first. Train a specialized head only if measured quality is insufficient.

This avoids paying for adaptation where zero-shot performance already meets requirements.

### What this slide does not claim

The architecture is the motivation and proposed system design. The notebook tests **one email taxonomy**, not independent serving across many teams.

The plan's single-RTX-3090 feasibility assessment and its performance estimates are planning inputs, not measured production results from the email evaluation.

**Suggested narration:** "We want each team to improve its routing without owning an entire language model. We share a frozen encoder, specialize small scoring heads, and measure whether zero-shot routing is already sufficient before training."

---

## Slide 2: A controlled test of head-only adaptation

### What was classified?

The notebook uses the Hugging Face dataset `jason23322/high-accuracy-email-classifier`, a synthetic, highly regular email benchmark.

The six categories are:

| Category | Intended meaning |
|---|---|
| `forum` | Forum discussions, replies, threads, and related community notifications. |
| `promotions` | Legitimate marketing, offers, sales, and commercial campaigns. |
| `social_media` | Social-network activity such as follows, likes, and mentions. |
| `spam` | Unsolicited, deceptive, suspicious, or low-quality bulk messages. |
| `updates` | Operational, account, service, order, billing, or other status updates. |
| `verify_code` | One-time passwords, login codes, and account-verification messages. |

Each email becomes a state containing **subject + body**, with one six-way `Choice` question.

This is a **closed-set task**: the system must select from those six categories. It does not establish that the model can recognize an email that belongs to none of them.

### How was the data divided?

| Partition | Rows | Purpose |
|---|---:|---|
| Fit | 8,624 | Input partition for supervised head training. |
| Development | 2,156 | Train-derived holdout for development evaluation and seed-stability checks. |
| Published test | 2,697 | Final reported comparison on the same emails before and after adaptation. |

The published training data contains **10,780 rows**, divided into the fit and development partitions using a fixed, stratified 80/20 split with seed 42.

**Stratified** means preserving approximately the same category proportions in each partition. The published test has about 449-451 emails per class.

There are two distinct validation layers:

1. The trainer reserves **10% of the fit IDs** for internal validation and checkpoint selection.
2. The separate **2,156-row development set** checks generalization during development.

Thus, 8,624 is the fit input partition size, not the number of examples used for gradient updates after the trainer's internal holdout.

The notebook checks that IDs do not overlap across the partitions. However, different IDs can contain identical text; slide 4 explains this limitation.

### What changed between the compared models?

| Held fixed | Changed |
|---|---|
| Qwen3-8B encoder | Released head versus email-specific trained head |
| Subject/body state representation | Learned state/action projections and logit scale |
| Six category labels | |
| Question instructions and candidate descriptions | |
| The same 2,697 test emails | |
| Temperature 1.0 | |

This makes the central comparison a test of **supervised head adaptation**, rather than a mixture of a new model, new labels, and a different test set.

### How was the head trained?

- **Warm start:** initialize from the released head rather than random weights.
- **Hard labels:** each training email has one target category.
- **Soft cross-entropy (`softce`):** train the six-way probability distribution to assign higher probability to the correct category.
- **Learning rate:** `5e-4`.
- **Batch size:** 256.
- **Weight decay:** `1e-4`, a regularization setting.
- **Maximum epochs:** 30. An epoch is one training pass through the optimization data.
- **Early stopping:** stop after five epochs without internal-validation-accuracy improvement.
- **Checkpoint selection:** retain the best internal-validation checkpoint; seed 42 selected epoch 12.

Three seeds, 42, 123, and 2026, were used for development stability. A **seed** controls reproducible random choices such as the internal split and batch order. Seed 42 was declared as the final test model rather than chosen by comparing test scores across seeds.

### Why is temperature held at 1.0?

Temperature rescales the scores before converting them into probabilities:

```text
probability for class k = exp(score_k / T) / sum_j exp(score_j / T)
```

For fixed scores and positive temperature, this changes how sharp the probabilities are, not which score is largest.

The original diagnostic at temperature 0.1 had very extreme probabilities and log loss **19.3741**. At temperature 1.0, baseline log loss improved to **3.2873**, but accuracy remained **26.88%**.

**Interpretation:** temperature addressed probability sharpness, not the wrong ranking of categories. The headline before/after results use temperature 1.0 on both sides.

**Suggested narration:** "We hold the encoder, request format, taxonomy, temperature, and test emails constant. We change the projection head through supervised training, and select its checkpoint using validation rather than final test performance."

---

## Slide 3: Head fine-tuning resolves the zero-shot class collapse

### 1. Start with the overall result

Both models successfully returned predictions for all **2,697** test emails.

| Measure | Released zero-shot head | Email-specific fine-tuned head | Desired direction |
|---|---:|---:|---|
| Top-1 accuracy | 26.88% | 97.59% | Higher |
| Macro F1 | 0.1399 | 0.9759 | Higher |
| Top-2 accuracy | 42.23% | 99.67% | Higher |
| Multiclass log loss | 3.2873 | 0.0707 | Lower |
| 10-bin expected calibration error (ECE) | 0.3406 | 0.0070 | Lower |
| Distinct classes predicted | 2 of 6 | 6 of 6 | All required classes represented |

These metrics answer different questions. Accuracy measures the first choice, macro F1 checks balanced class performance, top-2 accuracy checks the shortlist, log loss evaluates the full probabilities, and ECE checks how well stated certainty aligns with observed success.

### 2. Top-1 accuracy: how often was the first choice right?

**Definition**

```text
Top-1 accuracy = correctly classified emails / all evaluated emails
```

The predicted category is the category with the highest probability.

| Outcome | Zero-shot | Fine-tuned |
|---|---:|---:|
| Correct | 725 | 2,632 |
| Incorrect | 1,972 | 65 |
| Total | 2,697 | 2,697 |

Therefore:

```text
Zero-shot:  725 / 2,697 = 26.88%
Fine-tuned: 2,632 / 2,697 = 97.59%
```

**Plain-language interpretation:** the released head gets roughly 27 out of 100 benchmark emails right; the tuned head gets roughly 98 out of 100 right.

Those rounded "out of 100" descriptions summarize this benchmark. They are not forecasts for real enterprise traffic.

### 3. The +70.71 percentage-point improvement

```text
Accuracy gain = 97.5899% - 26.8817%
              = 70.7082 percentage points
              ~= 70.71 percentage points
```

A **percentage point**, abbreviated **pp**, is the difference between two percentages.

Do not describe this as a "70.71% relative improvement." Relative percentage change divides the difference by the starting value; that is a different calculation.

**Interpretation:** on this test set, the tuned model correctly routes about 71 more emails per 100 evaluated emails than the released head.

### 4. What does "class collapse" mean?

Here, **class** means a category, not a training session or model component.

The zero-shot model selected only two categories:

| Predicted category | Zero-shot count | Fine-tuned count |
|---|---:|---:|
| `forum` | 0 | 436 |
| `promotions` | 1,966 | 443 |
| `social_media` | 0 | 469 |
| `spam` | 0 | 458 |
| `updates` | 731 | 454 |
| `verify_code` | 0 | 437 |

All six categories occur in the true test labels. Nevertheless, the zero-shot model never selects four of them.

That is the observed **class collapse**: the decision behavior concentrates on a small subset of the available labels.

It does not mean the server failed or returned only two probability keys. The model returned distributions over all six options; only two ever ranked first.

The tuned head selects every category and has high recall for each. This is stronger evidence than merely showing that all six labels appear at least once.

### 5. Precision, recall, and F1: what does macro F1 summarize?

To understand macro F1, first consider one category, such as `spam`.

| Term | Question it answers |
|---|---|
| True positive (TP) | How many actual spam emails were correctly labeled spam? |
| False positive (FP) | How many non-spam emails were incorrectly labeled spam? |
| False negative (FN) | How many actual spam emails were incorrectly sent to another category? |
| Precision | Of the emails labeled spam, what fraction really were spam? |
| Recall | Of all actual spam emails, what fraction did the model find? |

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

**Precision** penalizes sending the wrong emails into a category. **Recall** penalizes missing emails that belong there.

**F1** is their harmonic mean, balancing both. It is not simply the arithmetic average of precision and recall.

For an actual tuned-model example, `forum` has:

- **450** true forum emails.
- **430** correctly identified forum emails.
- **436** total emails predicted as forum.
- **Precision:** `430 / 436 = 98.62%`.
- **Recall:** `430 / 450 = 95.56%`.
- **F1:** approximately **0.9707**.

**Macro F1** computes F1 independently for each of the six classes and averages those six values:

```text
Macro F1 = (F1_forum + F1_promotions + ... + F1_verify_code) / 6
```

Every category gets equal weight, regardless of its frequency.

**Why this matters here:** accuracy alone can hide ignored categories, especially with imbalanced data. The baseline's four zero-recall categories also have zero F1, pulling macro F1 down to **0.1399**.

After tuning, macro F1 reaches **0.9759**, showing that improvement is broad rather than confined to the two labels the baseline favored.

Macro F1 is a score on a 0-to-1 scale. A value of 0.9759 does **not** mean exactly 97.59% of emails are correct; that interpretation belongs to accuracy. The two happen to be numerically close for this balanced, high-performing result.

### 6. Top-2 accuracy: was the correct category at least in the shortlist?

**Definition**

```text
Top-2 accuracy =
    emails whose true category is among the two highest-scoring choices
    / all evaluated emails
```

- Zero-shot: **1,139 / 2,697 = 42.23%**.
- Fine-tuned: **2,688 / 2,697 = 99.67%**.

For the tuned model, there are 65 top-1 mistakes. Of those, **56** have the true category in second place; **9** do not have it in the top two.

**Interpretation:** the tuned model's two-category shortlist is almost always useful on this benchmark.

**What it does not mean:** automatic routing accuracy is not 99.67% if the system still chooses just one category. Someone or something would need to resolve the shortlist, and that second-stage process requires separate evaluation.

### 7. Multiclass log loss: did the probabilities favor the true answer?

Accuracy only asks whether the top choice was correct. **Log loss** also asks how much probability the model assigned to the actual category.

For one email:

```text
Loss = -ln(probability assigned to the true category)
```

The reported multiclass log loss is the average across emails, using the natural logarithm.

Illustrative single-email values:

| Probability assigned to the true label | Log loss |
|---:|---:|
| 0.99 | 0.010 |
| 0.90 | 0.105 |
| 0.50 | 0.693 |
| 0.01 | 4.605 |

These examples explain the metric; they are not individual notebook observations.

Giving the true category almost no probability creates a large penalty. Consequently, confident wrong predictions are particularly costly.

**Observed change:** **3.2873 -> 0.0707**.

**Interpretation:** after tuning, the probability distributions place much more mass on the correct categories and incur far fewer severe probability errors.

Log loss is not a percentage, an error rate, or a measure of milliseconds. Lower is better; zero is the ideal.

### 8. Expected calibration error: does stated certainty match actual success?

A classifier is **calibrated** when its probability estimates agree with observed correctness over groups of predictions.

For example, among many predictions made with approximately 90% top-label probability, a calibrated model should be right approximately 90% of the time.

This does not mean any one prediction is guaranteed to be right.

The reported **10-bin ECE**:

1. Takes each email's highest class probability.
2. Places it in one of ten probability ranges.
3. Computes each nonempty bin's average top probability and actual accuracy.
4. Takes the absolute difference in each bin.
5. Averages those differences, weighted by the number of emails per bin.

```text
ECE = sum over bins [
    fraction of examples in bin
    * abs(bin accuracy - bin mean top probability)
]
```

**Observed change:**

- Zero-shot: **0.3406**, approximately **34.06 percentage points** of weighted absolute bin-level discrepancy.
- Fine-tuned: **0.0070**, approximately **0.70 percentage points**.

**Interpretation:** on this benchmark, the tuned model's probability estimates align much more closely with its empirical success rate.

Important limits:

- ECE is not the fraction of wrong predictions.
- The absolute gap does not, by itself, distinguish overconfidence from underconfidence.
- It depends on the binning method and can hide class-specific or subgroup problems.
- Good ECE on synthetic test data does not establish calibration under real-world distribution shift.
- A low aggregate ECE does not eliminate individual confident mistakes.

#### CLM's `confidence` field is not the probability used for ECE

The API defines:

```text
CLM confidence = highest class probability
                 - mean probability of the other classes
```

For six probabilities summing to one:

```text
CLM confidence = p_max - (1 - p_max) / 5
```

For example, if the top probability is 0.90, the confidence margin is:

```text
0.90 - 0.10 / 5 = 0.88
```

The **reported 10-bin ECE uses the top probability**, not this margin. The notebook's earlier confidence-margin diagnostic is therefore not the same calculation as the final ECE result.

Neither the raw margin nor the top probability should be treated as a guaranteed probability of correctness for new traffic. Abstention thresholds need validation on representative data.

### 9. "All class recalls >= 95.56%": does the model miss an entire category?

Recall measures how much of a true category the model recovers.

| True category | Correct / total | Tuned recall |
|---|---:|---:|
| `forum` | 430 / 450 | 95.56% |
| `promotions` | 442 / 449 | 98.44% |
| `social_media` | 445 / 449 | 99.11% |
| `spam` | 442 / 449 | 98.44% |
| `updates` | 440 / 449 | 98.00% |
| `verify_code` | 433 / 451 | 96.01% |

**Interpretation:** no category is being ignored. Even the lowest-recall category, forum, recovers about 96% of its true examples.

This is not a claim that every class has 95.56% precision. Precision and recall answer different questions.

The largest remaining error pairs are:

- **15** true forum emails classified as social media.
- **15** true verification-code emails classified as spam.

These provide concrete targets for future error analysis.

### 10. "1,915 mistakes corrected; 8 new mistakes": why compare the same emails?

The comparison is **paired**: each email has both a baseline prediction and a tuned prediction.

| | Tuned correct | Tuned wrong | Total |
|---|---:|---:|---:|
| Baseline correct | 717 | 8 | 725 |
| Baseline wrong | 1,915 | 57 | 1,972 |
| Total | 2,632 | 65 | 2,697 |

This separates gains from regressions:

- **1,915 corrected:** wrong before, correct after.
- **8 regressions:** correct before, wrong after.
- **717 retained successes:** correct both times.
- **57 persistent errors:** wrong both times.

The net gain is:

```text
1,915 - 8 = 1,907 additional correct classifications
1,907 / 2,697 = 70.71 percentage points
```

**Interpretation:** the gain is not simply a different mixture of errors with little net benefit. Corrections greatly exceed newly introduced mistakes.

The notebook also reports an exact **McNemar test**, which compares the two disagreement counts, 1,915 and 8. Its displayed p-value is `0.0` because the value is too small for the numerical representation used; it is not literally zero.

This supports a clear paired difference on the evaluated benchmark. It does not address synthetic-data bias or production readiness.

### 11. The 95% bootstrap interval: how stable is the measured gain?

The reported interval for the paired accuracy improvement is:

> **+69.78 to +71.60 percentage points**, around the observed **+70.71 pp**.

**Bootstrap** means repeatedly resampling the observed test examples with replacement and recomputing the statistic. An example can appear multiple times in a resample while another is omitted.

The notebook's reported method is:

- **Paired:** baseline and tuned outcomes for the same email stay together.
- **Class-stratified:** resampling preserves the benchmark's class mix.

**Interpretation:** under this resampling procedure, the estimated gain remains very large rather than disappearing with ordinary variation in which benchmark examples are included.

A 95% confidence interval is not:

- A claim that every email has a 95% chance of being correct.
- An interval for the tuned accuracy itself.
- A guarantee of performance on future real emails.
- A measure of uncertainty from choosing a different training seed.

Its frequentist interpretation concerns repeated use of the interval-estimation procedure under its assumptions, rather than a literal 95% probability that a fixed true value lies in this already-computed interval.

**Important limitation:** row-level resampling does not account for all dependence between repeated templates or near-duplicate emails. The interval is conditional on this benchmark and its approximately balanced categories; it can understate uncertainty about genuinely new email styles or sources.

### 12. How the metrics fit together

| Result | What it establishes on this test | What it does not establish |
|---|---|---|
| High top-1 accuracy | Most first choices are correct. | Correctness on a different data distribution. |
| High macro F1 | Performance is strong across categories. | Equal performance for every real-world subgroup. |
| High top-2 accuracy | The true class is usually in the shortlist. | The same accuracy for automatic one-label routing. |
| Low log loss | Probability mass generally favors the truth. | Perfect calibration or absence of confident errors. |
| Low ECE | Binned probability estimates track observed success. | Reliable uncertainty on unseen or adversarial traffic. |
| Six classes predicted and high recall | The observed two-label collapse is resolved. | Recognition of unknown categories. |
| Large paired gain and narrow interval | The benchmark improvement is substantial and consistent. | A unique advantage over every simpler supervised classifier. |

**Suggested narration:** "The released head does not merely make scattered errors: it ignores four of the six categories. Fine-tuning restores all six and raises accuracy to 97.59%. Macro F1 shows the gain is broad across labels, while top-2 accuracy shows a strong shortlist. Lower log loss and ECE show the probabilities are also more useful. The paired counts and bootstrap interval support a large measured improvement, but all of that remains conditional on this synthetic benchmark."

---

## Slide 4: Strong benchmark result, not yet a production claim

### What evidence supports the result?

#### Stability across training seeds

Development accuracy across three seeds averages **98.42%**, with a sample standard deviation of **0.37 percentage points**.

**Standard deviation** summarizes how much the observed scores vary around their mean. Here, the small spread suggests the result is not dependent on one unusually favorable training run.

However, the outer fit/development split stays fixed. This is not a test of stability across different datasets, time periods, teams, or independently chosen outer splits.

#### Small development-to-test gap

The declared seed-42 model scores:

- **98.28%** development accuracy.
- **97.59%** published-test accuracy.
- A gap of approximately **0.69 percentage points**.

This is reassuring within the benchmark. Both partitions still come from the same synthetic source, so the gap does not measure real-world robustness.

### Why is duplicate content a concern?

Different IDs do not guarantee different examples.

After normalizing text, the notebook finds:

| Overlap with the fit partition | Published-test rows | Share |
|---|---:|---:|
| Exact subject-plus-body match | 229 | 8.5% |
| Exact body match | 1,379 | 51.1% |

Repeated content can make evaluation easier than deployment on genuinely new messages.

The notebook therefore evaluates subsets excluding exact matches:

- **97.61% accuracy** on 2,468 rows without an exact subject-plus-body match.
- **96.74% accuracy** on 1,318 rows without an exact body match.

These results indicate that exact duplicates alone do not explain the high score. They do not eliminate shared templates, paraphrases, or near-duplicates.

**Group deduplication** means grouping related examples, such as a shared template or thread, and keeping each group in only one partition. That is a stronger future control than checking IDs or exact text alone.

### Why include a simpler architecture control?

A logistic-regression classifier trained on the same frozen Qwen state embeddings reaches **94.29% development accuracy**.

This **linear probe** asks whether a simple supervised decision boundary on the existing embeddings already solves much of the task.

Its strong result suggests that:

- The frozen representation already contains useful classification information.
- Supervised adaptation is a major contributor to the improvement.
- This experiment does not prove CLM's projection architecture is uniquely necessary.

The linear probe's score is on **development data**, not the published test. It must not be directly ranked against the tuned model's 97.59% test score as if they were a matched architecture comparison.

### What must happen before production?

1. **Use representative real emails.** Include the team's actual categories, writing styles, languages, message lengths, and difficult cases, subject to privacy and access controls.
2. **Create stronger splits.** Group related templates or threads together and use time-aware evaluation to expose drift.
3. **Compare fair baselines.** Evaluate zero-shot CLM, tuned CLM, a frozen-embedding linear classifier, and conventional text-classification methods under one fixed protocol.
4. **Validate abstention.** Choose thresholds on validation data, then measure both the quality of accepted routes and how many emails require fallback.
5. **Test unknown and adversarial inputs.** Include messages outside the six labels and misleading or adversarial content.
6. **Measure operational performance.** Benchmark new-state latency, throughput, memory, multi-team isolation, rollback, and monitoring.

Two useful operational terms:

- **Coverage:** the share of incoming emails automatically routed after applying an abstention rule. This is different from successful-request coverage; the baseline's 100% successful evaluation does not mean a safe abstention policy was tested.
- **p95 latency:** the response time at or below which 95% of measured requests finish. It is not the average or median.

The separate notebook microbenchmark reports **warm-cache median** times of 1.5 ms for the baseline and 3.2 ms for the tuned model over the same 200 emails. These are not new-email p95 latency or end-to-end production measurements.

### Final interpretation

The evidence supports:

> A task-specific CLM projection-head checkpoint can turn the tested zero-shot formulation into a strong classifier for this in-distribution synthetic email task while keeping Qwen3-8B frozen.

It does not yet support:

> The system is production-ready, generalizes to all team taxonomies, or outperforms every simpler supervised alternative.

**Suggested narration:** "The right next step is real-data validation, not immediate production rollout. Seed stability and duplicate-aware subsets are reassuring, but synthetic templates and a strong linear baseline limit the claim. We now need representative data, fair comparisons, validated fallback rules, and operational testing."

---

## Source and result map

- [Experiment plan](./experiment.md): motivation, per-team architecture, proposed experiments, feasibility estimates, and rollout gates.
- [Evaluation notebook](../notebooks/clm_email_classification_evaluation.ipynb): data preparation, controlled training design, saved outputs, diagnostics, and interpretation.
- [Zero-shot metrics at temperature 1.0](../notebooks/artifacts/clm_email_metrics_zero_shot_t1.json): baseline quality, class counts, and classification report.
- [Fine-tuned metrics at temperature 1.0](../notebooks/artifacts/clm_email_metrics_finetuned_v1_t1.json): tuned quality, class counts, and classification report.
- [Paired comparison](../notebooks/artifacts/clm_email_pre_post_comparison.json): bootstrap intervals, paired correction counts, ECE, and development-seed statistics.
- [Zero-shot row-level predictions](../notebooks/artifacts/clm_email_predictions_zero_shot_t1.csv) and [fine-tuned row-level predictions](../notebooks/artifacts/clm_email_predictions_finetuned_v1_t1.csv): underlying outcomes and probabilities used to verify counts, top-2 results, and ECE.
