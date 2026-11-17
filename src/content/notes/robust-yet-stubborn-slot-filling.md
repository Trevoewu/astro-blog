---
title: "Robust Yet Stubborn: Why Advisor-Guided Zero-Shot Slot Filling Stalls on Annotation Conventions"
pubDate: 2026-11-17 19:00:00
description: "Why source-trained open-weight extractors override valid advice from frontier LLM advisors, what can be fixed without target labels, and how 1–3 confirmations per conflict pair settle the rest."
slug: robust-yet-stubborn-slot-filling
published: true
---

## Abstract

Zero-shot cross-domain slot filling faces a sharp asymmetry between two model classes. Frontier black-box large language models (LLMs) generalize broadly to unseen slots, yet they make frequent boundary errors and cannot be fine-tuned locally. Conversely, compact open-weight models can be fine-tuned to enforce strict span boundaries and output formatting, yet they lack broad coverage on unseen slots. **Advisor-guided slot filling** bridges this gap by pairing a frozen black-box LLM *advisor* with a source-trained open-weight *extractor*. Yet even when the advisor supplies an exact gold span on an unseen slot, the fine-tuned extractor frequently overrides it.

We study why this happens across SNIPS and ATIS and establish three findings. First, source fine-tuning makes the extractor stubborn very early in training, long before it finishes learning to reject noisy advice. This drop happens even on slot names the model already knows, so it is not caused by unfamiliarity with new slots. Second, these overrides are not random mistakes on rare slots. In ATIS, the extractor moves valid advisor spans into more general slots; in SNIPS, it repeatedly rejects a few common words—such as `"my"` for `playlist_owner`—because source training always left those words unlabeled. Third, this split explains what can and cannot be fixed without target labels. A simple slot-name check stops wrong-slot copying on ATIS with zero target labels, whereas no label-free method can tell whether a disputed word like `"my"` is valid under a new dataset's rules. Checking just one to three examples for each recurring disagreement settles those rules cleanly and avoids the heavy side effects of few-shot fine-tuning.

---

## 1. Introduction

Zero-shot cross-domain slot filling extracts target-domain slot values from user utterances without labeled target training data (Liu et al., 2020; Yu et al., 2021). In practice, practitioners face a sharp trade-off between two classes of models. **Frontier black-box large language models (LLMs)** generalize broadly to unseen target slots, yet they cannot be fine-tuned locally and often produce loose span boundaries when prompted directly. Conversely, **compact open-weight LLMs** can be fine-tuned cheaply on source domains to enforce strict span boundaries and output formatting, yet they lack broad coverage on unfamiliar target slots.

**Advisor-guided extraction** combines both strengths through a simple division of labor:

- The **advisor** is a frozen, black-box frontier LLM that reads the utterance and target schema to propose candidate spans for each slot. Because the advisor is never modified, it can be swapped freely across model providers.
- The **extractor** is a compact, open-weight LLM fine-tuned on source-domain data to make the final prediction from the utterance, schema, and the advisor's suggestions.

Across cross-domain benchmarks and advisor models, this hybrid setup consistently outperforms running either model alone.

Yet this division of labor creates an unexpected puzzle. Given the same advisor suggestions, an untuned base model is **controllable but gullible**: it accepts nearly all valid advice, but it also copies most of the advisor's mistakes. Fine-tuning on source domains flips the extractor into a **robust yet stubborn** state. It learns to filter out noisy advice, but it also begins overriding the advisor—even when the advisor supplies the exact gold span on an unseen slot.

This stubbornness shows up in two concrete ways. On SNIPS, the tuned extractor repeatedly drops valid words that clash with source-domain habits. In *"add this track to my global funk"*, for instance, `"my"` is the gold value for `playlist_owner`. Other SNIPS domains leave `"my"` unlabeled, so source training teaches the extractor to ignore it. At test time, the untuned base model accepts the advisor's suggestion `playlist_owner="my"` every time, whereas the source-tuned extractor rejects it every time. On ATIS, a second pattern appears: instead of dropping a valid advisor span, the tuned extractor moves it from a specific role slot (such as `fromloc.city_name`) into a generic slot (such as `city_name`).

Why does source fine-tuning make an extractor override valid advice, and when can we fix it without target labels? To track this shift, Section 2 measures two behaviors (adapting Li et al., 2022; Goyal et al., 2025): **conflict controllability** ($\mathcal{C}_{\mathrm{conf}}$), how often the extractor accepts valid advice when its own prediction is wrong, and **strict noise robustness** ($\mathcal{R}$), how often it keeps a correct prediction when the advice is wrong. Our study answers the question in three steps:

- **Stubbornness sets in early and is not caused by unfamiliar slot names (Section 3).** The extractor's willingness to follow conflicting advice drops within the first 200 training steps, long before noise rejection finishes improving. Scaling down the fine-tuned weights at inference time offers no middle ground where both behaviors stay strong. This drop happens equally on familiar and unfamiliar slots, and neither explicitly marking unseen slots nor selectively routing uncertain predictions makes the extractor defer to the advisor.
- **Overrides concentrate on a few recurring pairs taught by source training (Section 4).** Most conflicts are not random errors on rare slots; they repeat on a small set of slot-and-word pairs. In SNIPS, words like `"my"`, `"this"`, and `"here"` appear hundreds of times in source training with *zero* gold labels, teaching the extractor to reject them on sight. Swapping `"my"` for an ordinary name, or patching in a single late layer from the untuned base model, immediately restores compliance. Fine-tuning thus installs a targeted word-refusal habit rather than breaking the model's ability to use advice.
- **Wrong-slot copying can be fixed without labels, whereas word conventions need a few human confirmations (Sections 5–7).** Because ATIS wrong-slot copying moves spans between slots with the same base name (such as `fromloc.city_name` and `city_name`), a simple rule that routes hijacked spans back to the advisor's slot fixes it without target labels and without hurting SNIPS. By contrast, no label-free method—not even stronger LLM judges—can fix word omissions like `playlist_owner="my"`, because slot names alone do not reveal which words a dataset counts as valid. However, because these disputes repeat on a few pairs, asking an annotator to confirm just one to three examples per pair settles the rule cleanly and avoids the heavy side effects of few-shot fine-tuning.

---

## 2. Experimental Setup

### 2.1 Task, Benchmarks, and Models

#### Task and Slot Partition

An open-weight extractor $E_\theta$ is trained on labeled source domains with slot vocabulary $\mathcal{S}_{\mathrm{src}}$ and evaluated on a held-out target domain with schema $S = (s_1, \dots, s_m)$. Each slot $s_j$ has a canonical name and a brief description. Target slots split into **source-seen slots** ($S_{\mathrm{seen}} = S \cap \mathcal{S}_{\mathrm{src}}$) and **unseen target slots** ($S_{\mathrm{unseen}} = S \setminus \mathcal{S}_{\mathrm{src}}$). Given an utterance $x$ and schema $S$, $E_\theta$ predicts $\hat{y} = (\hat{y}_1, \dots, \hat{y}_m)$, where each $\hat{y}_j$ is either verbatim substring(s) from $x$ or `NONE`.

#### Benchmarks

We evaluate on two English benchmarks summarized in Table 2:

- **SNIPS** (Coucke et al., 2018): A 7-fold leave-one-domain-out benchmark covering 39 unique slots, with 10,984 test utterances and 500 validation utterances per fold. Source-seen slots are non-empty in all 7 folds, while unseen slots appear in 6 of 7 folds (all except `SearchCreativeWork`).
- **SNIPS-to-ATIS** (Hemphill et al., 1990): Extractors trained on SNIPS are tested zero-shot on the 83-slot ATIS flight-booking schema (5,371 test utterances with 17,720 gold entities, plus 500 validation utterances). ATIS shares 0 of 83 exact slot names with SNIPS, placing every slot in $S_{\mathrm{unseen}}$ and introducing dot-qualified role hierarchies such as `fromloc.city_name`, `toloc.city_name`, and `city_name`.

#### Frozen Advisors and Source-Tuned Extractor

At inference time, a frozen LLM *advisor* reads the utterance and target schema to propose candidate slot values, while the source-trained *extractor* $E_\theta$ conditions on those suggestions to produce the final prediction. To evaluate the extractor on its own (`Solo` mode), we run it without external advisor guidance.

We evaluate four frozen advisors spanning multiple model families, with `DeepSeek-V4-Flash` serving as our primary advisor. Unless noted otherwise, the extractor $E_\theta$ is a `Qwen3-8B` model (Yang et al., 2025) fine-tuned on source-domain utterances via LoRA across three random seeds.

### 2.2 Separating Controllability from Noise Robustness

For each slot $s_j$ with gold label $y_j$, let $\hat{y}^{\mathrm{solo}}_j$ and $\hat{y}_j$ denote the extractor's solo and advised predictions, and let $a_j$ denote the advisor's non-abstaining suggestion for slot $s_j$ (either a proposed candidate span or `NONE`). We call a suggestion *correct* if it matches the gold label ($a_j = y_j$), and *wrong* otherwise. This divides advised predictions into the four cases in Table 1.

| | **Advice correct** | **Advice wrong** |
| :--- | :--- | :--- |
| **Solo correct** | Redundant agreement | Harmful disagreement ($\mathcal{R}$) |
| **Solo wrong** | Advice-critical ($\mathcal{C}_{\mathrm{conf}}$) | Both wrong |

*Table 1: Four-cell breakdown of advisor–extractor interaction on a single slot. Conflict controllability ($\mathcal{C}_{\mathrm{conf}}$) isolates the advice-critical cell; strict noise robustness ($\mathcal{R}$) isolates the harmful-disagreement cell.*

Adapting Li et al. (2022), we measure the two off-diagonal behaviors:

$$
\mathcal{C}_{\mathrm{conf}} = \Pr\bigl(\hat{y}_j = y_j \;\big|\; \hat{y}^{\mathrm{solo}}_j \neq y_j,\; a_j = y_j\bigr) \tag{1}
$$

$$
\mathcal{R} = \Pr\bigl(\hat{y}_j = y_j \;\big|\; \hat{y}^{\mathrm{solo}}_j = y_j,\; a_j \neq y_j\bigr) \tag{2}
$$

Two measurement choices keep stubbornness visible:

1. **Isolating true conflicts from easy agreements.** Averaging over all correct suggestions mixes genuine conflicts with easy cases where the extractor already agrees with the advisor. On a 700-utterance SNIPS sample (`DeepSeek-V4-Flash`, seed 3407), 86.6% of all correct suggestions survive in the final output, which hides the fact that the extractor follows conflicting suggestions only 48.36% of the time on SNIPS validation.
2. **Enforcing strict span robustness.** Equation (2) requires the extractor to keep the exact gold value $y_j$, not merely reject a bad suggestion. In our training-checkpoint and weight-scaling analyses, we compute $\mathcal{C}_{\mathrm{conf}}$ and $\mathcal{R}$ over suggestions proposing candidate spans and evaluate absent-slot suggestions separately, preventing trivial empty agreements from inflating robustness.

### 2.3 Evaluation Regimes and Zero-Label Boundaries

| Study Component | Scope & Size | Target Labels Used? | Protocol Role |
| :--- | :--- | :--- | :--- |
| SNIPS main & uncertainty routing | 7 folds, 39 slots (10,984 test; 700-utt sample) | None at inference / routing | Diagnostic analysis |
| ATIS error flow & gate design | 83 slots (5,371 test, 17,720 gold; 1,000-utt sample) | None at inference / gating | Exploratory design |
| ATIS untouched gate validation | 500 untouched ATIS val utts, `V4-Flash`, 3 seeds | None (eval-only gold) | **Confirmatory test** (3/3 seeds pass) |
| Training & weight-scaling dynamics | SNIPS 500 val/fold; SNIPS 700 / ATIS 1,000 / `AddToPlaylist` 500 | Eval-only (diagnostic) | Diagnostic analysis |
| Zero-label cluster verification | SNIPS 500 val → 10,984 test; ATIS 2,685 / 2,686 split | None (zero target labels) | **Confirmatory test** (all 6 fail) |
| Minimal cluster confirmation | Same SNIPS (23 span) & ATIS (8/13/8) cluster splits | 1–3 yes/no labels / cluster | **Confirmatory test** |
| Few-shot fine-tuning comparison | SNIPS seed 3407 (80 target utts, 605 slots / 238 spans) | 80 fully labeled utterances | **Confirmatory test** & error audit |

*Table 2: Overview of experimental splits, target-label access, and evaluation protocols across the study.*

Because mixing target supervision into zero-shot evaluation blurs what unlabeled adaptation can actually achieve, we enforce a strict boundary across three regimes (Table 2):

- **Strict zero-label regime (Sections 3–5).** Target-domain inputs consist strictly of slot schemas and unlabeled utterances. Recurring $(\mathrm{slot}, \mathrm{span})$ conflict clusters are discovered solely from disagreements between `Solo` and advisor outputs on unlabeled text: on SNIPS validation (500 utterances per fold), we retain span clusters with at least 15 pairs and slot clusters with at least 30 pairs; on ATIS, we split the 5,371 test utterances evenly into 2,685 unlabeled discovery utterances (retaining clusters with at least 81 pairs) and 2,686 disjoint evaluation utterances. No target gold labels ever enter cluster discovery, zero-label judge verification, or structural gating.
- **Separating exploratory design from confirmatory validation.** Whenever an analysis or rule is informed by inspecting model errors, we validate it on an untouched split with locked evaluation criteria. Specifically, while our training-step sweep in Section 3 tracks checkpoints 100–2600 as a training-dynamics diagnostic, the structural rebinding gate designed on ATIS test errors is tested out-of-sample on 500 untouched ATIS validation utterances using freshly generated advisor suggestions.
- **Minimal-label verification vs. few-shot fine-tuning (Sections 6–7).** Target gold labels enter *only* when we test how much supervision is needed to break convention deadlocks: either 1–3 binary (`yes`/`no`) confirmations per recurring conflict cluster, or 80 fully annotated target utterances for few-shot fine-tuning.

---

## 3. The Stubbornness Paradox and What Does Not Explain It

Given identical advisor suggestions, why does source fine-tuning turn a compliant base model into a stubborn extractor? Before examining training data priors, we establish the behavioral contrast between the base and fine-tuned models and rule out four intuitive explanations: late-stage overfitting, overly strong adapter weights at inference time, unfamiliarity with new slot names, and selective uncertainty routing.

### 3.1 Base Model vs. Source-Tuned Extractor

| Dataset (Sample) | Model | Format Fail | Strict F1 | Correct Kept | Wrong Removed | Wrong-Slot Copies / New FP |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| SNIPS (700 utts) | Base (`Solo`) | 0.0% | 54.46 | — | — | — |
| | Base (`+V4-Flash`) | 0.0% | 69.65 | 98.6% (1,298/1,316) | 7.5% (49/654) | 10 / 18 (55.6%) |
| | Source-Tuned (`Solo`) | 0.0% | 75.05 | — | — | — |
| | Source-Tuned (`+V4-Flash`) | 0.0% | 79.39 | 86.6% (1,140/1,316) | 80.4% (526/654) | 22 / 80 (27.5%) |
| ATIS (1,000 utts) | Base (`Solo`) | 5.5% | 12.63 | — | — | — |
| | Base (`+V4-Flash`) | 0.3% | 79.54 | 97.5% (2,859/2,931) | 24.6% (324/1,316) | 3 / 26 (11.5%) |
| | Source-Tuned (`+V4-Flash`, 3-seed) | 0.0% | 64.90 | 79.4% (76.2%–82.3%) | 60.6% (58.0%–65.0%) | 88.2%–92.6% (953–1,119 / 1,080–1,208) |

*Table 3: Untuned `Qwen3-8B` vs. the source-tuned extractor on frozen `DeepSeek-V4-Flash` suggestions. On ATIS, the untuned base model without advice exceeds a 5% format-failure rate (55/1,000). Source fine-tuning flips the extractor from gullible to robust yet selectively stubborn, producing convention omission on SNIPS and schema-bias slot reassignment on ATIS.*

Table 3 contrasts untuned `Qwen3-8B` against the source-tuned extractor on diagnostic samples from SNIPS (700 utterances) and ATIS (1,000 utterances) using identical `DeepSeek-V4-Flash` suggestions. The untuned base model is **controllable but gullible**: it preserves 98.6% of correct suggestions on SNIPS and 97.5% on ATIS, yet filters out only 7.5% and 24.6% of wrong suggestions, respectively. Moreover, without advisor guidance (`Solo`), the untuned model fails to produce valid structured output on 5.5% of ATIS utterances and reaches only 12.63 F1.

Source fine-tuning eliminates formatting errors and shifts the extractor into a **robust yet stubborn** state. On SNIPS, fine-tuning lifts advised F1 to 79.39 and rejects 80.4% of wrong suggestions, but correct-suggestion retention drops to 86.6%. On ATIS—where every slot is unseen—the tuned extractor still filters out 60.6% of wrong suggestions across seeds, yet strict F1 falls to 64.90, far below the untuned base model's 79.54.

Inspecting the overridden predictions reveals that stubbornness takes two distinct structural forms:

1. **Clustered convention omission (SNIPS):** When a target domain's annotation rules clash with source-domain habits, the tuned extractor suppresses valid advisor spans to `NONE`. Across all 32 occurrences of `playlist_owner="my"` in the SNIPS sample, the untuned base model accepts 32/32 (100%), whereas the source-tuned extractor accepts 0/32 (0%).
2. **Schema-bias slot reassignment (ATIS):** Rather than dropping unfamiliar spans, the tuned extractor moves them to a different slot. On the ATIS sample, 88.2%–92.6% of new false positives across seeds copy a valid advisor span from a specific role-qualified slot into a generic fallback slot (such as moving `fromloc.city_name` into `city_name`). Across the full 5,371-utterance ATIS test set, where raw `V4-Flash` predictions achieve 77.60 F1, the tuned extractor drops to 62.18–66.92 F1, with 91.6% of all lost entities (16,333/17,823 pooled) reappearing under another slot name.

### 3.2 Early Collapse During Fine-Tuning (Not Late Overfitting)

![Training-checkpoint trajectory on SNIPS target validation](./assets/controllability_trajectory_20261111.png)

*Figure 1: Training-checkpoint trajectory on SNIPS target validation (seed 3407, 7 folds, candidate-span advice). Between steps 100 and 200, conflict controllability $\mathcal{C}_{\mathrm{conf}}$ collapses in parallel on both unseen (64.50% → 38.43%) and seen (66.82% → 46.24%) target slots, while strict noise robustness $\mathcal{R}$ climbs toward its plateau.*

A natural first hypothesis is that stubbornness comes from training too long on source domains, so early stopping should avoid it. Figure 1 rules this out by tracking conflict controllability ($\mathcal{C}_{\mathrm{conf}}$) and strict noise robustness ($\mathcal{R}$) from step 100 to step 2600 across all 7 SNIPS folds. Between steps 100 and 200, $\mathcal{C}_{\mathrm{conf}}$ plunges from **65.27%** to **41.02%** and stays flat for the rest of training (averaging 41.52% across steps 200–2400, and 48.36% at validation-selected checkpoints). Meanwhile, strict noise robustness $\mathcal{R}$ is still climbing from **77.18%** at step 100 to **82.33%** at step 200 before reaching its **88.44%** plateau.

Three checks confirm that this early collapse is a genuine behavioral shift rather than a shifting-sample artifact:

- **Fixed conflict population:** As `Solo` accuracy improves after step 100, the total number of conflict instances shrinks by only 16%, whereas accepted conflicts drop by 46%. Restricting evaluation to a fixed set of slot instances that remain in conflict across *every* checkpoint reproduces the exact same drop between steps 100 and 200 (60.17% → 48.44%).
- **Symmetry across seen and unseen slots:** Between steps 100 and 200, $\mathcal{C}_{\mathrm{conf}}$ falls in lockstep on source-seen slots (**66.82%** → **46.24%**) and unseen target slots (**64.50%** → **38.43%**). Because every training update occurs on source-seen slots, this parallel drop reflects a general shift toward trusting parametric predictions over external context, echoing context-parametric inversion in instruction tuning (Goyal et al., 2025).
- **Immediate suppression on prior-conflicting domains:** On `AddToPlaylist`, where `playlist_owner="my"` contradicts source training, $\mathcal{C}_{\mathrm{conf}}$ falls from 28.20% at step 100 to 13.78% at step 200 and 8.05% at step 1800.

### 3.3 Weight Scaling Reveals No Middle Ground

![Linearly scaling the LoRA adapter weight lambda on SNIPS and ATIS](./assets/adapter_dial_20261111.png)

*Figure 2: Linearly scaling the LoRA adapter weight $\lambda \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ on SNIPS (700 utts) and ATIS (1,000 utts), seed 3407. Stubbornness turns on sharply between $\lambda=0.25$ and $\lambda=0.5$ before wrong-advice rejection $\mathcal{R}$ matures.*

If early stopping cannot separate noise rejection from stubbornness, can we find a sweet spot at inference time by scaling down the LoRA adapter weight $\lambda \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$?[^1] Figure 2 shows that across both benchmarks, stubbornness switches on sharply between $\lambda=0.25$ and $\lambda=0.5$, well before noise robustness matures:

- **SNIPS (700 utterances):** As $\lambda$ increases from $0.0$ to $1.0$, fixed-span $\mathcal{C}_{\mathrm{conf}}$ falls from 96.30% to 49.38% while strict $\mathcal{R}$ climbs from 7.40% to 89.00%. Crucially, compliance on `playlist_owner="my"` collapses from 32/32 ($\lambda=0.0$) and 26/32 ($\lambda=0.25$) to **1/32** at $\lambda=0.5$ (and 0/32 for $\lambda \ge 0.75$), even though noise robustness at $\lambda=0.5$ still lags the full adapter by 27.2 points (61.80% vs. 89.00%). The same cliff appears on the full 500-utterance `AddToPlaylist` validation split, where `"my"` compliance drops from 99/119 at $\lambda=0.25$ to **1/119** at $\lambda=0.5$.
- **ATIS (1,000 utterances):** Wrong-slot copies stay low at $\lambda=0.0$ (3) and $\lambda=0.25$ (69), then surge to **740** at $\lambda=0.5$ and 1,008 at $\lambda=0.75$, pulling advised F1 down from 80.22 ($\lambda=0.25$) to 71.44 ($\lambda=0.5$) and 66.68 ($\lambda=1.0$). Meanwhile, dialing $\lambda \le 0.25$ reintroduces severe `Solo` format failures (5.6% at $\lambda=0.25$ and 11.7% at $\lambda=0.0$) and sacrifices SNIPS F1 (75.29 vs. 79.48). No adapter scale $\lambda$ preserves formatting stability and noise rejection without triggering stubbornness.

### 3.4 Unfamiliar Slot Names Are Not the Cause

A third intuitive explanation is that the extractor overrides the advisor on unseen slots simply because it does not recognize the new slot names, and would defer to the advisor if told explicitly which slots were unseen during training. We tested this directly in a controlled holdout study: we held out 10 source-seen slots as simulated unseen targets and compared a standard source-tuned extractor against one trained with an explicit unseen-slot familiarity marker.

If unfamiliarity with unseen slot names drove the problem, the extractor should perform poorly on those held-out slots when run without advice (`Solo`). Instead, `Solo` accuracy on the 10 held-out slots reached **95.34%** in the standard model and **94.32%** in the familiarity-marked model—already hitting 87.4% by step 50. Adding explicit unseen-slot markers improved unseen conflict controllability by only $+2.05$ percentage points (95% CI $[-1.31, +5.74]$) and left `playlist_owner="my"` compliance at 2.5% (vs. 14.1% in the unmarked control). The extractor does not override the advisor because unseen slot names are opaque; it overrides the advisor when source-trained priors actively oppose the suggested span.

### 3.5 Selective Uncertainty Routing Cannot Bypass Stubbornness

| Routing Policy (Budget) | F1 | Fixed | Broken |
| :--- | ---: | ---: | ---: |
| `Solo` (0%) | 75.05 | 0 | 0 |
| Full advice (100%) | 79.39 | 214 | 87 |
| Oracle (10%, 530 asked) | 79.44 | 136 | 0 |
| Oracle (25%) | 80.00 | 195 | 11 |
| Token confidence (10%) | 76.27 | 67 | 30 |
| Token confidence (25%) | 77.34 | — | — |
| Token confidence (50%) | 77.40 | — | — |
| Unseen-first (50%) | 78.24 | 138 | 33 |

*Table 4: Inference-time uncertainty routing on the 700-utterance SNIPS benchmark (`DeepSeek-V4-Flash`, seed 3407). Oracle routing at 10%–25% matches or exceeds full advice, and token confidence provides a useful query-ranking signal (AUROC 0.744 vs. 0.4376 for prompted self-report), but routing cannot overcome stubbornness once the advisor is queried.*

Finally, could we avoid conflicts at inference time by querying the advisor *only* when the extractor is uncertain? Evaluating selective routing on the 700-utterance SNIPS benchmark (Table 4) shows that output token confidence is a much better uncertainty signal than prompted self-report: asking the model whether it needs help yields an AUROC of only 0.4376 before generation (asking *less* often when wrong: 28.1% vs. 37.8% when right) and 0.472 after generation, whereas output token confidence reaches an AUROC of 0.744 (0.717 on `NONE` predictions). Under an oracle routing policy that queries only the top 10%–25% most beneficial slots, F1 reaches 79.44–80.00, matching or beating 100% full advice (79.39) at a fraction of the query cost.

Yet uncertainty routing is purely an efficiency tool, not a cure for stubbornness. Among the 530 slots queried under 10% oracle routing, 394 remain unfixed—and on **167** of those unfixed slots, the advisor actually supplies the exact gold answer (146 gold spans and 21 valid absent-slot predictions), which the extractor still rejects. Because the extractor is highly confident in its source-trained priors, unseen F1 on `AddToPlaylist` stays near zero (1.45–4.30) regardless of the routing budget.

---

## 4. Root Cause: Source-Trained Priors and Convention Clusters

Having ruled out late overfitting, adapter scaling, slot unfamiliarity, and routing, we now examine *what* the extractor is actually rejecting. Disagreements between the extractor and advisor are not random errors scattered across rare slots. Instead, they concentrate heavily in a small number of recurring `(slot, span)` clusters where target-domain annotation conventions clash with strong negative priors learned during source training.

### 4.1 Conflicts Concentrate in a Few Recurring Clusters

Cross-domain benchmarks frequently differ on annotation boundary policies that cannot be deduced from a single utterance alone: whether a possessive pronoun like `"my"` counts as a `playlist_owner`, whether a deictic word like `"this"` fills `object_select`, whether `"here"` fills `current_location`, or whether generic nouns like `"flight"` and `"after"` fill ATIS slots. Grouping unlabeled disagreements between `Solo` and the advisor by `(slot, span)` and `(slot)` reveals extreme concentration across both benchmarks:

- **SNIPS:** Across all 7 validation folds (`DeepSeek-V4-Flash`, seed 3407), just **52** recurring clusters—23 specific `(slot, span)` pairs (occurring $\ge 15$ times, covering 1,088 instances) and 29 broader `(slot)` clusters ($\ge 30$ times)—account for **3,068 of 4,585** (66.9%) total conflicts. The same concentration holds for `DeepSeek-V3.2` (51 clusters covering 2,940/4,720 conflicts) and `Qwen3-27B` (34 clusters covering 668 span-cluster instances).
- **ATIS:** On the 2,685-utterance discovery split, an average of just **9.67** recurring `(slot, span)` clusters per seed (8, 13, and 8 clusters across the three seeds) cover an average of 2,211.3 conflict instances. Five recurring word clusters shared across all three seeds—(`depart_time.time_relative`, `"after"`), (`flight`, `"flight"`), (`flight`, `"flights"`), (`transport_type`, `"flight"`), and (`transport_type`, `"flights"`)—alone account for **69%** of all recoverable oracle F1 gain ($+1.95$ out of $+2.81$ F1).

### 4.2 Source Training Actively Teaches the Opposite Rule

| Target Slot | Span | Source Occurrences (Gold Labels) | Target Validation Gold Rate |
| :--- | :--- | ---: | ---: |
| `playlist_owner` | `"my"` | 375 (0 gold) | 99.2% (119/120) |
| `object_select` | `"this"` | 624 (0 gold) | 97.2% (106/109) |
| `current_location` | `"here"` | 65 (0 gold) | 95.2% (20/21) |

*Table 5: Source-training frequency versus target-domain gold rate for representative SNIPS conflict clusters (`DeepSeek-V4-Flash`). The extractor saw these exact words hundreds of times in source domains with zero gold labels, learning a strong negative prior that directly contradicts the target domain's annotation convention.*

Why does the fine-tuned extractor refuse these specific clusters so stubbornly? Checking how often these exact target words appeared in the six SNIPS source-training domains reveals that the extractor is not merely unfamiliar with them—it was repeatedly trained to leave them unlabeled (Table 5).

In the source domains for `AddToPlaylist`, the word `"my"` appears **375 times** across 372 training utterances with **0** gold slot labels. Source fine-tuning therefore teaches the extractor a strong negative prior: whenever `"my"` appears in an utterance, leave it out. When transferred to `AddToPlaylist`, where `"my"` is the gold span for `playlist_owner` **99.2%** of the time (119/120 validation instances), this learned rule collides head-on with the target convention. The exact same contradiction explains `"this"` for `object_select` in `RateBook` (**624** source occurrences with **0** gold labels vs. **97.2%** target gold) and `"here"` for `current_location` in `GetWeather` (**65** source occurrences with **0** gold labels vs. **95.2%** target gold).

### 4.3 Training Oscillation and Mechanistic Localization

To verify that source fine-tuning installs a targeted lexical refusal rather than breaking the model's general ability to copy advisor spans, we examined both how the `"my"` refusal emerges across training checkpoints and how it is represented inside `Qwen3-8B`.

#### Competitive Oscillation Before Convergence

Tracking the 119 `AddToPlaylist` validation utterances where the advisor correctly proposes `"my"` for `playlist_owner` shows that the extractor does not forget `"my"` smoothly. Instead, compliance oscillates sharply as source supervision competes against the base model's prior: after accepting `"my"` at step 100 (`Solo` 84/119, advised 27/119), compliance drops to 0/119 across steps 200–600, rebounds to roughly half of validation instances at steps 700 (56/119) and 900 (69/119), and finally locks into near-zero compliance from step 1200 onward (0/119 at selected checkpoint 1800).

#### Causal Localization to Late-Layer Lexical Priors

Probing the extractor's next-token probabilities on `playlist_owner="my"` conflicts ($n=707$) confirms that this converged stubbornness is a narrow word-level suppression:

- **Base vs. fine-tuned next-token prior:** On actual conflict prompts, untuned `Qwen3-8B` assigns `"my"` as its top-1 next token in **99.72%–99.86%** of cases, whereas the source-tuned extractor drops to **0.0%** across all three seeds.
- **Counterfactual span substitution:** If we keep the exact same utterance, slot name, and advisor suggestion format but replace the word `"my"` with an ordinary phrase (`"daphne s"`, $n=696$), the source-tuned extractor immediately resumes following the advisor, raising its top-1 acceptance rate from 0.0% to **56.32%–96.12%** across seeds. The extractor's advice-following circuit is intact; only the word `"my"` is blocked.
- **Single-layer activation patching:** Patching the residual stream activation from a single late layer (layer 34 out of 36) of the untuned base model into the source-tuned extractor at the target slot token ($n=273$ per seed) restores `"my"` as the top-1 next token in **95.24%–98.53%** of cases (reaching **98.90%–100.0%** at layers 35–36). Source fine-tuning thus acts as a localized late-layer gate that vetoes specific words penalized during source training.

### 4.4 Why Blanket Overrides Hurt Overall

| Benchmark (Scope) | Rule | $\Delta$F1 | Fixed : Harmed | Cluster Acc. |
| :--- | :--- | ---: | ---: | ---: |
| SNIPS (all 52 clusters) | `V-always` | $-2.77$ | 1,996.7 : 4,939.7 | 27.2% |
| SNIPS (23 span clusters only) | `V-always` | $+0.74$ | 1,129.0 : 1,050.7 | — |
| ATIS (29 clusters / 3 seeds) | `V-always` | $-2.14$ | 166.7 : 1,042.0 | 28.8% |

*Table 6: Blanket zero-label overrides across recurring conflict clusters (`DeepSeek-V4-Flash`). Because genuine target conventions coexist in clusters alongside systematic advisor mistakes, blindly forcing the extractor to follow conflicting advice degrades overall F1.*

If conflicts concentrate in a few recurring clusters, why not simply force the extractor to follow the advisor whenever a disagreement falls into one of those clusters? Table 6 shows why blanket override rules fail: **recurring conflict clusters contain both genuine target conventions and recurring advisor hallucinations.**

Alongside valid target conventions like `playlist_owner="my"` or `object_select="this"`, the advisor also repeatedly proposes invalid spans that violate target guidelines—such as tagging generic words like `"weather"` for `condition_description` or `"restaurant"` for `restaurant_type` on SNIPS, or attaching spurious modifiers on ATIS. Consequently, forcing the extractor to accept all 52 recurring SNIPS clusters (`V-always`) lowers overall F1 by **$-2.77$** (fixing 1,996.7 instances but harming 4,939.7; 27.2% cluster accuracy). On ATIS, `V-always` across the discovered clusters drops F1 by **$-2.14$** (166.7 fixed vs. 1,042.0 harmed). Even if we restrict `V-always` to the 23 specific `(slot, span)` clusters on SNIPS, it harms nearly as many predictions as it fixes ($1{,}129.0 : 1{,}050.7$, $+0.74$ F1) and severely damages domains like `GetWeather` ($-4.31$ F1). Resolving stubbornness therefore requires separating valid convention clusters from recurring advisor errors.

---

## 5. What Zero-Label Methods Recover

Having established that advisor–extractor conflicts stem from clustered prior mismatches rather than isolated utterance noise, we now ask what can be recovered strictly without target-domain labels. Comparing structural output de-biasing against label-free cluster heuristics reveals a sharp divide: **structural constraints on schema slot names reliably repair wrong-slot copying on hierarchical schemas, whereas uniform label-free heuristics cannot resolve word-level convention disputes.**

### 5.1 Zero-Label Structural De-Biasing: Same-Family Span-Binding Gate

On hierarchical schemas such as ATIS, where source-trained extractors exhibit schema-bias slot reassignment (Section 3), a deterministic structural output gate can repair hijacked spans using only schema slot names and current advisor suggestions—requiring neither target labels nor retraining.

#### Mechanism

Define the slot-family head $h(s)$ as the final dot-delimited segment of a slot name $s$ (e.g., $h(\texttt{fromloc.city\_name}) = h(\texttt{city\_name}) = \texttt{city\_name}$). Whenever the extractor outputs a non-empty value $v$ on slot $s_k$ for which the advisor proposed no candidate span, while the advisor proposed $v$ on a distinct same-family slot $s_j$ ($h(s_j) = h(s_k)$) that the extractor left `NONE`, the **Same-Family Span-Binding Gate** blocks the hijacked prediction on $s_k$ and rebinds $v$ to $s_j$. Note that when $S_{\mathrm{seen}} = \varnothing$ (as on ATIS), a seen-slot gate cannot fire, making structural rebinding the primary zero-label lever.

#### Why Family Restriction Is Mandatory

Although the gate was originally formulated *post hoc* from ATIS test error flow (seed 3407, 5,371 utterances, `DeepSeek-V4-Flash`), restricting rebinding to same-head families is essential for cross-domain safety. As shown in Table 7, unconstrained cross-family variants push ATIS seed 3407 even higher—from 66.91 baseline ($\mathrm{TP}=13{,}371$, $\mathrm{FP}=8{,}879$, $\mathrm{FN}=4{,}349$) to 81.10 for Full Bind ($\mathrm{TP}=15{,}023$, $\mathrm{FP}=4{,}303$), 81.09 for Role-Family Bind, and 81.94 for Family+Drop (vs. 75.79 for Block hijack alone, $\mathrm{FP}=4{,}002$, and 75.70 for Drop gate)—but they do so only by overfitting ATIS's zero-seen-slot topology. On SNIPS (10,984 utterances × 3 seeds × 3 advisors, `Qwen3-27B` seed 5203 omitted; *Descriptive* transfer check), Full Bind *lowers* F1 for every advisor: `V4-Flash` drops $79.69 \to 78.47$, `V3.2` drops $78.60 \to 76.64$, and `Qwen3-27B` drops $79.68 \to 77.53$, while Family+Drop drops further to 77.19 / 75.72 / 75.93. By contrast, Same-Family Span-Binding preserves a $+11.55$ F1 gain on ATIS seed 3407 ($66.91 \to 78.46$; $\mathrm{P}=74.91$, $\mathrm{R}=82.35$, $\mathrm{TP}=14{,}593$, $\mathrm{FP}=4{,}887$, $\mathrm{FN}=3{,}127$; 2,938 utterances changed), surpassing raw advisor suggestions (77.60), while guaranteeing an exact 0.00 side effect on flat schemas such as SNIPS (79.69 / 78.60 / 79.68).

| Gate Variant | ATIS (3407) | SNIPS V4 | SNIPS V3.2 | SNIPS Qwen |
| :--- | ---: | ---: | ---: | ---: |
| Base (ungated) | 66.91 | 79.69 | 78.60 | 79.68 |
| Raw advisor suggestions | 77.60 | — | — | — |
| Block hijack only | 75.79 | — | — | — |
| Drop gate | 75.70 | — | — | — |
| Full Bind | 81.10 | 78.47 | 76.64 | 77.53 |
| Role-Family Bind | 81.09 | — | — | — |
| Family + Drop | 81.94 | 77.19 | 75.72 | 75.93 |
| **Same-Family Bind** | **78.46** | **79.69** | **78.60** | **79.68** |

*Table 7: Ablation of Span-Binding Gate variants on ATIS full test (seed 3407, 5,371 utts, Post hoc) and SNIPS transfer check (10,984 utts × 3 seeds × 3 advisors, Descriptive). Only Same-Family Bind preserves ATIS gains above raw advisor suggestions while guaranteeing 0.00 side effect on SNIPS.*

#### Independent Reproduction and Clean Out-of-Sample Validation

On the independent 2,686-utterance ATIS evaluation half (`Option B1`, `V4-Flash`, *Pre-set rule* baseline), Same-Family Span-Binding reproduces a $+13.03$ F1 gain across all 3 seeds ($65.20 \to 78.23$ mean; seed 3407: $66.72 \to 77.89$, seed 4101: $61.75 \to 78.67$, seed 5203: $67.13 \to 78.14$).

More importantly, in a clean out-of-sample validation on 500 untouched ATIS validation utterances with freshly generated zero-shot `DeepSeek-V4-Flash` advice (*Pre-set rule*), the unchanged Same-Family Span-Binding Gate passes the pre-set $\ge +5.0$ F1 threshold in **3/3 seeds**:

- Seed 3407: $68.36 \to 78.30$ (**$+9.94$** F1, 95% CI $[8.80, 11.14]$, 253/500 utterances changed);
- Seed 4101: $62.23 \to 77.12$ (**$+14.89$** F1, 95% CI $[13.28, 16.55]$, 320/500 changed);
- Seed 5203: $66.89 \to 77.81$ (**$+10.92$** F1, 95% CI $[9.81, 12.11]$, 297/500 changed).

Across the three seeds, mean F1 rises from 65.83 to 77.74 (**$+11.92$** F1; paired `Solo` mean 47.19), again exceeding raw advisor suggestions (75.85). Extending across 18 additional secondary ATIS runs spanning multiple advisors and evaluation splits (*Descriptive*), Same-Family Span-Binding improves **21/21 ATIS runs total** by $+9.43$ to $+21.72$ F1.

### 5.2 Ceiling of Zero-Label Cluster Heuristics

Finally, we test whether uniform label-free heuristics can resolve clustered convention omissions on SNIPS without target labels. We evaluate fixed rules over the 23 SNIPS span clusters (`V4-Flash`, 3 seeds; *Post hoc* subset in Stage 0, replicated as a *Pre-set rule* baseline in Stage 1): `V-always` (always adopt the cluster's advisor span) and `V-vote` (adopt when the advisor agrees by majority vote). On the 23 span clusters, these heuristics yield only modest gains with low precision: `V-always` adds $+0.74$ F1 (80.43, $1{,}129:1{,}050.7 = 1.07:1$ fixed:harmed) and `V-vote` adds $+0.98$ (80.67, $923.3:632.3 = 1.46:1$). Across `V3.2` (`V-vote` $+1.29$) and `Qwen3-27B` (`V-vote` $+1.84$), fixed:harmed ratios remain poor (1.07:1–2.00:1 on `V4-Flash` and `V3.2`).

When applied without subsetting or across datasets (*Pre-set rule* baselines), uniform zero-label heuristics fail outright. Across all 52 SNIPS clusters (span and slot clusters combined), every heuristic loses F1: `V-always` drops $-2.77$ and `V-vote` drops $-0.37$. On ATIS (29 clusters across 3 seeds), `V-always` drops $-2.14$ F1 ($166.7:1{,}042 = 0.16:1$). Zero-label heuristics fail because they have no signal to separate valid target conventions (such as `"my"` or `"this"`) from recurring advisor mistakes (such as `"weather"` or `"flights"`).

---

## 6. The Identifiability Study

Because uniform zero-label heuristics cannot distinguish valid target conventions from recurring advisor errors, we next test whether *any* label-free verdict source can identify cluster conventions—and, if not, how many target labels are strictly required to make them identifiable.

### 6.1 Stage 1 Strict Zero-Label Verification Test

#### Setup and Pre-Set Criteria

We evaluate six purely label-free verdict sources on the fixed SNIPS (23 span clusters, `DeepSeek-V4-Flash`) and ATIS (29 clusters across 3 seeds: 8/13/8, `DeepSeek-V4-Flash`) setups (*Pre-set rule*):

1. `J-self`, the same advisor LLM (`DeepSeek-V4-Flash`) judging cluster conventions from bare slot descriptions and unlabeled target utterances;
2. `J-cross-glm` (`GLM-5.3-Flash`) and
3. `J-cross-haiku` (`Claude-Haiku-5-5`), cross-family LLM judges;
4. `B`, the untuned base extractor (`Qwen3-8B`) queried directly for cluster verdicts;
5. `L`, a Stage 1a source-training span-gold lookup; and
6. `E`, the primary conservative ensemble requiring multi-source agreement (and abstaining otherwise).

Pre-set pass criteria on SNIPS `V4-Flash` require an F1 gain $\ge +1.25$, a fixed:harmed ratio $\ge 5.0$, and cluster-weighted accuracy $\ge 80\%$ (on ATIS: $\ge +1.4$ F1 and $\ge 5:1$).

#### Pre-Set Outcome: "Nothing Passes"

As shown in Table 8, ensemble `E` fails every criterion on SNIPS ($-0.13$ F1 [$-0.1295$], $387.3:659 = 0.59:1$, 28.0% accuracy, 4 follow / 8 reject / 11 abstain), and **none** of the five individual sources passes SNIPS `V4-Flash`. (On ATIS, `E` scores $+1.14$ F1 and $298.7:54.7 = 5.46:1$, but is not evaluated under the pre-set rule because primary SNIPS failed.)

| Source | SNIPS $\Delta$F1 | SNIPS Fixed:Harmed | SNIPS Ratio | SNIPS Acc / Verdicts | ATIS $\Delta$F1 | ATIS Fixed:Harmed | ATIS Ratio | ATIS Verdicts |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- |
| `J-self` (`V4-Flash`) | $+0.02$ | 942.0:1,367.3 | 0.69:1 | 49.7% (20F/3R) | $-1.22$ | — | 0.23:1 | 26F/3R |
| `J-cross-glm` | $+0.97$ | 1,022.0:778.7 | 1.31:1 | 69.6% (18F/5R) | $-2.14$ | — | 0.16:1 | 29F/0R |
| `J-cross-haiku` | $+1.69$ | 1,167.3:513.7 | 2.27:1 | 80.5% (18F/5R) | $+1.23$ | 338.3:160.3 | 2.11:1 | 22F/7R |
| `B` (base `Qwen3-8B`) | $+0.00$ | 441.3:666.3 | 0.66:1 | 54.3% (3F/20R) | $+2.19$ | 605.7:54.7 | 11.08:1 | 14F/15R |
| `L` (source lookup) | $-0.12$ | 193.0:212.7 | 0.91:1 | 24.0% (2F/6R/15A) | $-0.63$ | 178.0:170.0 | 1.05:1 | 13R/16A |
| **`E` (ensemble, primary)** | **$-0.13$** | **387.3:659.0** | **0.59:1** | **28.0% (4F/8R/11A)** | **$+1.14$** | **298.7:54.7** | **5.46:1** | — |

*Table 8: Stage 1 strict zero-label verdict evaluation (Pre-set rule). On SNIPS, neither ensemble `E` nor any individual source passes all three criteria ($\Delta\text{F1} \ge +1.25$, ratio $\ge 5:1$, accuracy $\ge 80\%$): "Nothing passes." Verdicts list Follow (F), Reject (R), and Abstain (A) cluster counts.*

#### Systematic Bias Decomposition

Every label-free source fails for a structural reason:

- **LLM judges default to `follow`:** `J-self` marks 20/23 SNIPS clusters and 26/29 ATIS clusters `follow`, and `J-cross-glm` marks 18/23 SNIPS and 29/29 ATIS `follow`. Even the strongest frontier judge, `J-cross-haiku`, passes the gain ($+1.69$) and accuracy (80.5%) thresholds on SNIPS (18/23 `follow`) and gains $+1.23$ on ATIS (22/29 `follow`), yet **fails the $5:1$ safety gate on both datasets** (2.27:1 on SNIPS; 2.11:1 on ATIS). Bare slot names and unlabeled utterances cannot tell a model which plausible interpretations an annotation team actually excluded.
- **Untuned base model `B` defaults to `reject`:** `Qwen3-8B` outputs `reject` on 20/23 SNIPS clusters (3/23 `follow`), yielding $+0.00$ F1 (0.66:1, 54.3% accuracy) on SNIPS while accidentally meeting the ATIS thresholds ($+2.19$, 11.08:1, 95.0% accuracy via 14 `follow` / 15 `reject`)—confirming sensitivity to prompt priors rather than convention knowledge.
- **Source lookup `L` contradicts target conventions:** `L` scores $-0.12$ on SNIPS (15/23 abstains, 24.0% accuracy) and $-0.63$ on ATIS (16/29 abstains, 23.5% accuracy) because target-specific convention words absent from source slot annotations (`"my"`, `"this"`, `"here"`) are labeled `reject`.

Secondary SNIPS advisor checks confirm the same failure: on `DeepSeek-V3.2` ($V_0 = 78.59$, *Pre-set rule*), all sources fail (`J-self` $+0.31$ [0.83:1], `J-cross` [`V4-Flash`] $-0.29$ [0.54:1], `B` $+0.13$ [0.74:1], `L` $-0.24$ [0.62:1], `E` $+0.67$ [1.81:1]). On `Qwen3-27B` ($V_0 = 79.30$, *Descriptive*), `J-self` returned HTTP 404 (`not run`), leaving `E` incomplete while `J-cross` ($+1.05$, 1.50:1), `B` ($+0.66$, 1.17:1), and `L` ($-0.28$, 0.29:1) all fall short.

### 6.2 Identifiability via Minimal Target Labels

To test whether annotation ambiguity, rather than cluster representation, is the sole missing ingredient, `Option B` replaces label-free verdicts on the exact same clusters with just $k=1$ or $k=3$ binary (`yes`/`no`) labels per span cluster (*Pre-set rule*).

| Dataset | Setting / Arm | Labels | $\Delta$F1 | Fixed:Harmed |
| :--- | :--- | ---: | ---: | ---: |
| SNIPS (`V4`) | `B-slot` (slot-level) | 69 | $+0.32$ | — |
| SNIPS (`V4`) | `B-rand` (random) | 69 | $+1.54$ | — |
| SNIPS (`V4`) | `V-k1` (cluster) | 23 | $+2.47$ | 54.9:1 |
| **SNIPS (`V4`)** | **`V-k3` (cluster)** | **69** | **$+2.50$** | **71.0:1** |
| SNIPS (`V4`) | `V-oracle` | 1,088 | $+2.50$ | — |
| ATIS (`V4`) | `V-k1` (ungated) | — | $+2.58$ | 21.4:1 |
| ATIS (`V4`) | `V-k3` (ungated) | 29 | $+2.78$ | 55.0:1 |
| ATIS (`V4`) | `gate only` | 0 | $+13.03$ | — |
| **ATIS (`V4`)** | **`gate + V-k3`** | **29** | **$+15.49$** | **74.9:1** |
| ATIS (`V4`) | `gate + V-oracle` | — | $+15.53$ | 268.0:1 |

*Table 9: Identifiability with minimal binary labels (`Option B1/B2`, Pre-set rule; `gate + V-k3/oracle` Descriptive). Just 1–3 yes/no answers per cluster recover virtually all oracle convention gain at $>50:1$ safety.*

- **SNIPS (`Option B2`):** With 69 total yes/no answers ($k=3$ across 23 span clusters, `V4-Flash`), `V-k3` recovers $+2.50$ F1 (82.19, $1{,}182.5:16.7 = 71.0:1$), matching the full 1,088-label `V-oracle` ($+2.50$, 82.19). Even a single binary label per cluster (`V-k1`, 23 labels) already captures $+2.47$ F1 (82.16, 54.9:1). Under pre-set `B2` targeting controls at the identical 69-label budget, `V-k3` outperforms both random-example labeling `B-rand` ($+1.54$ F1; margin $+0.96$, 95% CI $[0.25, 1.71]$) and slot-level allocation `B-slot` ($+0.32$ F1; margin $+2.17$, 95% CI $[1.86, 2.57]$). This advantage replicates on `DeepSeek-V3.2` ($+1.94$ vs. $+1.28$ / $+0.35$, 72 labels across 24 clusters) and `Qwen3-27B` ($+2.13$ vs. $+1.59$ / $+0.16$, 36 labels across 12 clusters).
- **ATIS (`Option B1`):** On the 2,686-utterance ATIS evaluation half (`V4-Flash`), 29 binary answers across 3 seeds ($k=3$ on mean 9.67 clusters) lift the ungated baseline ($V_0 = 65.20$) by $+2.78$ F1 to 67.97 ($647.6:11.8 = 55.0:1$, **PASS**), compared with $+2.58$ for `V-k1` (67.78, $639.3:29.9 = 21.4:1$) and $+2.81$ for `V-oracle` (68.01, $649:8 = 81.1:1$; `V-oracle` $-$ `V-always` $= +4.95$, **PASS**). When stacked on top of the Same-Family Span-Binding Gate (`gate only` $= 78.23$; *Descriptive*), `gate + V-k3` reaches **80.69** F1 ($+15.49$ over ungated $V_0$, $+2.46$ over `gate only`, $536.0:7.2 = 74.9:1$ rule fixes:harms), nearly matching `gate + V-oracle` (80.73, $536:2$).

### 6.3 Secondary Extension: Span-to-Utterance Propagation

Finally, `Option C` tests whether confirmed cluster verdicts from `S = V-k3` can propagate to utterances where the advisor did not propose the span (*Pre-set rule*). Rule `C1` (extending cluster verdicts to `Solo`-value disagreements) fails on SNIPS `V4-Flash` ($+0.003$ F1 over `S`, $4.6:2.6 = 1.74:1$, $+6$ labels; **FAIL**) and is ruled out (gating out its ATIS check of $+0.44$, $125.3:0.33 = 376:1$). Rule `C2` (propagating confirmed `follow` clusters such as `"my"` when the advisor missed or omitted the span in other utterances containing `"my"`) passes pre-set SNIPS rules ($+1.53$ F1 over `S` to 83.71, $790.4:24.5 = 32.2:1$, $+12.0$ labels $\to 81$ total; also $+1.82$ [$939:29.5$, $+18$ labels] on `V3.2` and $+0.47$ [$287.3:9$, $+6$ labels] on `Qwen3-27B`). However, `C2` completely fails to fire on ATIS ($+0.00$ F1, 0 active oracle clusters; **FAIL**), confining `C2` to a SNIPS-specific secondary finding rather than a general mechanism.

---

## 7. Limitations

### Comparison with 80-Utterance Few-Shot Fine-Tuning (`B3`, Audit, and Paused `B4`)

To contextualize the label-efficiency of cluster-level identifiability, `B3` compares the `S+C2` cluster-rule baseline (`M2`, 81 binary labels) against few-shot supervision using 80 labeled target utterances on SNIPS (`DeepSeek-V4-Flash`, seed 3407, Draw 0; *Pre-set rule*). From $V_0 = 79.52$ F1, few-shot extractor fine-tuning (`F-ext`) gains $+6.36$ F1 (85.88, 5,382 fixed / 2,753 harmed), beating `M2` ($+4.44$ F1 $\to 83.96$, 2,097 fixed / 44 harmed) on raw macro F1 (`M2` $-$ `F-ext` $= -1.92$, 95% CI $[-2.33, -1.50]$), triggering the pre-set `"baseline ahead"` stop rule after Draw 0 (with `M` without `C2` at $+2.88$, 1,295:20, and few-shot advisor prompting `F-adv` at $+3.80$, 2,314:463). Running `M2` on top of `F-ext` adds only $+0.04$ (85.92).

A detailed `Q1–Q4` audit of Draw 0 (*Descriptive*) exposes the cost–safety trade-off behind this headline F1:

1. **Redundancy after fine-tuning:** `F-ext` learns to emit `"my"` directly on 764/781 gold `"my"` test pairs (where `M2` fixes 776/781 on $V_0$), bypassing `M2`'s `output=NONE` precondition.
2. **Severe collateral harms:** `F-ext`'s 2,753 harmed slots split into 1,333 seen ($\Delta\mathrm{TP} = -41$, 1,538 fixed vs. 1,333 harmed) and 1,420 unseen ($\Delta\mathrm{TP} = +2{,}899$, 3,844 fixed), compared with `M2`'s 44 total harms (1 seen + 43 unseen). On `AddToPlaylist` (6 training utterances), `F-ext` causes 288 `artist` $\to$ `entity_name` and 283 `playlist` $\to$ `entity_name` swaps, gaining $+1.76$ F1 vs. `M2`'s $+12.44$; on `SearchCreativeWork` (3 utterances), `F-ext` drops $-1.68$ vs. `M2`'s $+0.26$. Most of `F-ext`'s headline gain comes from `SearchScreeningEvent` (30 utterances, $+28.06$ vs. $+7.98$).
3. **Annotation burden mismatch:** Full utterance annotation requires 80 utterances, 605 slot-presence decisions, and 238 gold spans, compared with `M2`'s 81 binary yes/no answers (69 `S` + 12 `C2`: 7.5× fewer binary decisions and 0 span annotations).
4. **Post-hoc seen-slot locking does not make `F-ext` harmless (`B4` paused):** Locking seen slots on `F-ext` Draw 0 (*Post hoc* local CPU check) lowers F1 by $-0.33$ ($+6.36 \to +6.03$, 95% CI $[-0.658, -0.011]$) while leaving all 1,420 unseen harms untouched (0 seen fixed/harmed; 1,085 dual seen+unseen same-span assignments), motivating pausing GPU execution of `B4`.

### Minimal Schema Descriptions

Both benchmarks provide terse slot descriptions in `rcsf.json` (*Descriptive*): SNIPS's 39 slots use 1–2 words (median 1, mean 1.49; 4–19 characters, median 10, mean 9.33; e.g., `"owner"` for `playlist_owner`, `"weather"` for `condition_description`, `"this current"` for `object_select`), while ATIS's 83 slots use 1–6 words (median 5, mean 3.55; 2–29 characters, median 23, mean 18.75). Richer annotation guidelines written prior to inspecting target errors could reduce convention ambiguity, though not eliminate dataset-specific boundary conventions.

### Two English Datasets Only

Our findings rest on SNIPS (7-fold leave-one-domain-out across crowd-sourced intents) and ATIS (single flight-booking domain with dot-qualified role hierarchies). Non-English schemas and multi-turn dialogue state tracking benchmarks (such as MultiWOZ) remain untested.

### Single 8B Extractor Architecture

Although we evaluate four distinct frozen advisors (`DeepSeek-V4-Flash`, `DeepSeek-V3.2`, `Qwen3-27B`, and `Luna xhigh`) and two cross-family judges (`GLM-5.3-Flash` and `Claude-Haiku-5-5`), the fine-tuned extractor is always `Qwen3-8B`. Whether larger extractors or alternative regularization schemes alter the early $\mathcal{C}_{\mathrm{conf}}$–$\mathcal{R}$ trade-off is left to future work.

---

## 8. Related Work

- **Zero-shot cross-domain slot filling.** Early cross-domain slot filling transfers to unseen target domains by conditioning sequence taggers on slot descriptions (Bapna et al., 2017; Lee & Jha, 2019) or example values (Shah et al., 2019), with context-aware representations (Siddique et al., 2021), decoupled span detection and typing (Liu et al., 2020), and contrastive slot sharpening (He et al., 2020; Zhang & Zhang, 2023) improving generalization. Other formulations cast slot filling as question answering or machine reading comprehension (Du et al., 2021; Yu et al., 2021; Liu et al., 2022; Liu et al., 2025), generative prompting (Yan et al., 2022; Li et al., 2023), or retrieval-augmented demonstration matching (Song et al., 2025). Prompted LLMs alone struggle with strict span boundaries compared to dialogue state tracking (Pan et al., 2023); staged prompting (Zhu et al., 2024), instruction tuning (Mirza et al., 2024), teacher distillation (Rana et al., 2025), and synthetic data generation (Huang et al., 2025) narrow this gap. In parallel, schema-driven dialogue modeling (Rastogi et al., 2020; Lin et al., 2021; Zhao et al., 2022) establishes schema-conditioned interfaces for cross-domain extraction.
- **Small–large model collaboration.** Recent work pairs compact extractors with larger LLMs in complementary roles. In GRSF (Li et al., 2025), an LLM generates slot knowledge filtered by a trained evaluator for a small tagger; in SynAug (Li et al., 2025) and SuperICL (Xu et al., 2024), a small model's predictions are injected into an LLM prompt that makes the final call. Other hybrids let a small model route uncertain intents to an LLM (Chen & Wang, 2025), learn question-generation strategies for zero-shot QA (Tavares et al., 2023), follow an LLM guidance prefix (Yao et al., 2024), or use reinforcement learning to decide when a 7B NER model should invoke external search (Chen et al., 2026). Closest in terminology, Asawa et al. (2026) train a small RL advisor to steer a frozen black-box LLM with free-form advice. Our setting reverses this asymmetry: a frozen LLM advisor provides candidate slot-level guidance to a source-trained 8B extractor, allowing us to isolate how supervised fine-tuning governs the extractor's willingness to follow or override external advice.
- **Controllability, noise robustness, and context-parametric inversion.** Balancing external context against parametric priors is a central tension in conditional generation. Li et al. (2022) formalize *controllability* (following counterfactual or conflicting context) versus *robustness* (ignoring irrelevant context) and propose Knowledge-Aware Fine-Tuning (KAFT) to preserve both. Similarly, RAFT (Zhang et al., 2024) and Yoran et al. (2024) train models on mixtures of oracle and distractor passages so retrieval-augmented generators ignore noisy context. However, Goyal et al. (2025) show that instruction fine-tuning exhibits *context-parametric inversion*: context reliance rises early in training and then collapses as non-context-critical examples dominate gradient updates. We observe an analogous dynamic in advisor-guided slot filling: source SFT makes the extractor highly robust to noisy advice while collapsing controllability on conflicting spans, yielding a robust-yet-stubborn regime where both source-prior convention omissions and schema-bias slot reassignments emerge.
- **Guideline refinement and interactive weak supervision.** Several recent frameworks address schema ambiguity through annotation guidelines or weak-supervision feedback. GoLLIE (Sainz et al., 2024) trains IE models to follow detailed Python-formatted annotation guidelines, assuming those target guidelines are already authored and available in the prompt. DiZiNER (Kim & Yoon, 2026) simulates pilot annotation across multiple LLMs and refines zero-shot NER instructions from inter-model disagreements without target labels. Interactive weak supervision (Boecking et al., 2021) queries human annotators on candidate labeling-function heuristics rather than individual examples. Our convention-conflict setting differs fundamentally from all three: target slot-filling schemas supply only brief descriptions without target-specific convention rules, inter-model disagreement cannot identify which convention a target benchmark actually follows (causing zero-label resolution to fail in Section 6), and minimal binary cluster-level verification resolves this identifiability gap with orders-of-magnitude fewer annotations than utterance-level few-shot fine-tuning.

[^1]: A deterministic replay check confirmed exact same-code reproducibility ($T = 0.000000$ across all cells), though minor endpoint differences ($0.07$–$0.32$ F1) relative to earlier batch runs left the pre-set endpoint gate undetermined; we therefore treat Figure 2 as a descriptive reference curve.
