# Countdown RLVR experimental protocol

**Status:** frozen protocol with completed measurements. The measured outcomes are in [the results report](../results/report.md); this file preserves the rules that were fixed before test evaluation.

## Research question and claim boundary

Test whether direct reinforcement learning with a shaped, exactly verifiable reward improves one-shot exact solution accuracy on synthetic Countdown arithmetic problems for a small instruction model. The result supports a claim about this task and protocol only. It does not establish general reasoning ability or show that an LLM is preferable to an exact solver.

The main comparison is direct GRPO against the original instruction checkpoint. A validity-only reward ablation separates admissible expression generation from reaching the target. An independent SFT adapter provides a supervised reference point and explicitly has access to solution witnesses.

## Fixed components and prompt

| Component | Frozen choice |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| License reference | Apache 2.0 per the pinned model card |
| Adaptation | LoRA; original model parameters frozen |
| RL library / method | Hugging Face TRL / GRPO with `loss_type=dr_grpo` |
| Main reward | `1.0` for an exact solution, `0.2` for a valid but incorrect expression, and `0.0` for invalid output |
| Validity ablation | `1.0` for an admissible expression using the supplied numbers; otherwise `0.0` |
| Main seeds | `42`, `43`, `44` |
| Compute cap | 72 accumulated GPU-hours; no paid compute or services |

The initial engineering smoke test used binary exact reward. Its first update group had all-zero rewards and zero gradient; a later phase produced rare exact-reward positives. Before any learning-rate pilot or final test evaluation, freeze the shaped reward above so valid expressions can provide a verifiable intermediate signal more often. Keep that binary-reward diagnostic in the local audit trail, separate from the final experiment matrix, and do not use test data to change this choice. The smoke run uses a fixed 16-row, arity-balanced validation subset to check checkpointing and adapter reload; pilot and final checkpoint selection use the frozen 128-row subset.

“Direct GRPO” means no project-specific SFT before RL. The public checkpoint itself is already instruction-tuned.

Use the tokenizer chat template with this fixed system message:

```text
You solve arithmetic puzzles.
Use every supplied number exactly once.
Use only +, -, *, / and parentheses.
Intermediate fractions and negative values are allowed.
Return a single arithmetic expression and nothing else.
```

The user message format is `Numbers: [2, 3, 7]\nTarget: 20`. Do not alter the prompt after pilot training begins.

## Task rules and verifier

Each problem has three or four positive integers (duplicates allowed) and an integer target. A completion must be one expression using every supplied number exactly once, respecting multiplicity. Only `+`, `-`, `*`, `/`, and parentheses are allowed. Intermediate negatives and fractions are allowed; division by zero is invalid. The final value must equal the target exactly.

Reject extra constants, decimal/scientific notation, unary operators, powers, functions, variables, attributes, strings, booleans, equality signs, Markdown, explanations, and multiple expressions. Strip surrounding whitespace, allow internal whitespace, and reject more than 256 characters.

The interface is `verify_expression(text: str, nums: list[int], target: int) -> Verification`. `Verification` contains `valid`, `correct`, `error_code`, `value_numerator`, and `value_denominator`. Parse with `ast.parse(..., mode="eval")`, allow only integer constants and supported binary nodes, and evaluate with `fractions.Fraction`. Never execute generated code. Error codes: `empty`, `too_long`, `invalid_characters`, `invalid_syntax`, `unsupported_node`, `wrong_numbers`, and `division_by_zero`. A valid expression that misses the target has `valid=True`, `correct=False`, and no error code.

## Synthetic data and leakage controls

All data is generated locally with seed `20261013`; each split has equal counts of three- and four-number puzzles.

| Split | Rows | Number range | Use |
|---|---:|---:|---|
| Train | 4,096 | 1–30 | GRPO prompts and SFT examples |
| Validation | 512 | 1–30 | Checkpoint selection and final validation report |
| Test ID | 1,024 | 1–30 | Primary held-out evaluation |
| Test OOD | 512 | 31–60 | Separate range-shift evaluation |

Enumerate multisets with `combinations_with_replacement`, then shuffle each arity using `random.Random(20261013 + arity)`. For each multiset, dynamic programming over input-position subsets enumerates all exact `Fraction` results from binary trees, both operand orders for subtraction and division, skipping division by zero. Keep reachable integer targets from 10 through 100, discard multisets with none, and select uniformly from each multiset's sorted reachable targets. Allocate distinct multisets to train, validation, then ID test; generate OOD independently. Shuffle displayed number order from a SHA-256-derived row-ID seed, interleave arities, and shuffle each partition deterministically.

This samples targets uniformly among reachable targets per multiset, not uniformly over all possible multiset-target pairs. Document that distribution. Save one fully parenthesized witness, selected by shortest character length and then lexicographic order. Public rows contain `id`, `split`, `nums`, `target`, and `arity`; store witnesses separately as `id` and `reference_expression`. GRPO must never read witnesses; SFT may read train witnesses only. Hash every split, witness file, prompt, and configuration before training. Do not tune or regenerate after test results are inspected.

## Training protocol

### Environment and hardware gate

Use Windows, Python 3.11, PyTorch 2.8.0 with CUDA 12.8 wheels, BF16, SDPA attention, and `trust_remote_code=False`. The target is an RTX 4070 Mobile; measure VRAM rather than assuming its amount. Before downloads or training, verify CUDA visibility, BF16 support, a small CUDA matrix operation, and at least 50 GB free disk. Record GPU, VRAM, driver, CUDA/PyTorch/library versions, trainable parameter count, and peak memory. If CUDA is unavailable, do CPU-side work and stop training; do not alter drivers/settings. Do not add vLLM, Unsloth, FlashAttention, Triton, DeepSpeed, or bitsandbytes.

LoRA settings: rank 16, alpha 32, dropout 0, no bias, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Freeze base weights and record the trainable parameter count.

| GRPO setting | Value |
|---|---:|
| Per-device batch / gradient accumulation | 1 / 16 |
| Generation batch / generations | 4 / 4 |
| Prompt / completion maximum | 256 / 96 tokens |
| Temperature / top-p / top-k | 1.0 / 1.0 / 0 |
| Iterations / loss / reward scaling | 1 / `dr_grpo` / `none` |
| KL beta / epsilon | 0.01 / 0.2 |
| Optimizer / scheduler | `adamw_torch` / constant |
| Warmup / weight decay / max grad norm | 0 / 0 / 1.0 |
| Gradient checkpointing | Enabled; `use_reentrant=false` |
| BF16 / truncation mask / vLLM | true / false / false |
| Evaluation / logging / workers / reporting | `no` / every 10 / 0 / none |

Four completions are generated for one prompt at a time; accumulation produces 16 completions per optimizer update (four prompts). Confirm grouping in smoke test. Do not set `steps_per_generation` with `generation_batch_size`; TRL derives it. Use the original model with the adapter disabled as the KL reference, without a second model copy. Dr. GRPO centers group rewards and uses a fixed loss denominator; log the rate of groups with no reward variation.

Use the same fixed curriculum for every training run: first 25% of optimizer updates uses three-number train problems only; the remaining 75% uses the full balanced train split. At the transition, reload the adapter and start a fresh optimizer with the same constant learning rate. Log the restart. GRPO never sees witness labels.

### Smoke test, pilots, and final runs

The smoke test uses eight steps on fixed train data and the 16-row validation subset described above. Check finite losses/rewards/gradients, grouping, at least one non-uniform reward group among up to 128 train problems, changed LoRA parameters after a signal-bearing update, unchanged sampled base weights, save/reload/resume, peak memory, and step time. On OOM, automatically rerun with generation batch 2 and two generations while keeping accumulation 16, save that sole permitted runtime override, and use it across experiments. If it still fails, stop and record the diagnosis; do not change model, precision, reward, or adaptation method.

Run two 100-step pilots from the original checkpoint, both seed 101 and the fixed curriculum. Pilot A uses LR `1e-4`; B uses `3e-4`. Before training, freeze 128 validation problems balanced by arity and selected by row-ID hash. Evaluate greedily; choose higher exact accuracy, then fewer invalid responses, then lower LR. Do not reuse pilot adapters.

| Experiment | Initialization | Seeds | Purpose |
|---|---|---|---|
| Original model | Pinned public checkpoint | — | Baseline |
| Main GRPO | New LoRA from public checkpoint | 42, 43, 44 | Shaped exact-correctness reward |
| Validity ablation | New LoRA from public checkpoint | 42 | Binary validity-only reward |
| SFT control | New LoRA from public checkpoint | 42 | Train on exact train witnesses |

SFT trains only on reference expression and EOS tokens, masking system/user/padding labels and with packing disabled. Use batch 1, accumulation 4, LR `1e-4`, same LoRA and curriculum, and same optimizer-update count/four problems per update. This approximately matches problem exposure, not compute or token count. Report its label access and cost.

Reserve up to 6 hours for baseline, smoke, and pilots; 56 hours for definitive training and validation; 10 hours for final evaluation/reporting. Time GRPO steps, validation generation, and eight SFT steps during pilots. Choose the largest of 1,200, 800, or 400 steps per experiment that fits the 56-hour allocation for three main seeds, ablation, SFT, validation, and a 20% margin. Freeze the count before final runs. If 400 does not fit, run one main seed plus ablation and SFT at 400 steps and mark the study single-seed. If that does not fit, fit only the seed-42 main GRPO run; if even that does not fit, stop with a feasibility report. Never exceed 72 GPU-hours or buy compute. Each stage has a cumulative ledger cap and leaves a 15-minute shutdown reserve.

Save and evaluate checkpoints every 100 steps, at phase changes, and at completion on the frozen 128-row validation subset. Select by exact accuracy, then fewer invalid responses, then fewer cumulative steps; include the original model as a candidate. Evaluate the selected checkpoint once on all 512 validation rows without reselection during final evaluation. After 200 steps of the first main run, continue seeds 43 and 44 only if a checkpoint improves at least 3 percentage points over baseline on the 128-row subset and at least 5% of measured reward groups contain a positive reward with within-group variation. Otherwise finalize the best checkpoint at or before step 200 as an explicitly budget-stopped single-seed result, then run controls allowed by the calibrated budget. Do not add SFT before RL, change rewards, or expand search. The stopping rule bounds cost but cannot guarantee improvement.

## Evaluation, uncertainty, and success target

Freeze all configurations and selected checkpoints before test. Evaluate the baseline, each completed main seed, ablation, and SFT on all 1,024 ID and 512 OOD rows with one greedy response (`do_sample=False`, `max_new_tokens=96`, batch 1), no tools, retries, or grammar constraints. Primary metric: exact solve rate. Also report validity, accuracy conditional on validity, scores by arity, each verifier error, generated tokens, truncation, latency, training time, and peak memory.

Separately sample four completions on fixed balanced subsets of 256 ID and 128 OOD rows (`temperature=1.0`, `top_p=1.0`, `top_k=0`, max 96 tokens). Keep all samples and report mean sampled pass-at-1 and pass-at-4; never mix them with greedy results.

Use 10,000 paired problem-level bootstrap resamples with seed `20261014` for improvement over baseline. With three main seeds, show every seed, mean and standard deviation across runs, plus a bootstrap interval for mean per-problem improvement. Explain that the interval captures problem variation, while seed variation is separate.

The pre-registered target was at least 10 percentage points mean improvement on ID test, a 95% paired bootstrap interval above zero, and no main seed below baseline. It was not met: the sole primary seed changed ID test accuracy by +0.59 pp (95% paired bootstrap interval −0.39 to +1.56 pp), and the validation continuation gate stopped seeds 43 and 44. The result is a one-seed primary experiment. Beating SFT was not required. OOD is reported separately and is not a success condition. Never tune after observing test.

## Artifacts and completion

Each unique run records `run_manifest.json` (config, model revision, Git commit, seeds, data hashes, hardware, dependencies, elapsed GPU-hours, reward-group statistics, failures), `training_metrics.jsonl`, `rollouts.jsonl` (ID, phase, step, raw completion, reward, verifier result), `checkpoint_evaluations.jsonl`, checkpoint files, and the selected adapter. Evaluation writes resumable per-split `predictions_*.jsonl` and `metrics_*.json` files with raw response, token count, verifier result, latency, finish reason, error counts, and arity metrics. Retain failed runs; never reuse run IDs.

Version the small synthetic data, configs, selected manifests, compact training metrics and checkpoint evaluations, final predictions, metrics, and figures. Ignore local weights, checkpoints, environments, and caches. Keep the selected adapter reloadable locally; prepare the seed-42 main adapter for a `v0.1.0` GitHub release with checksum and inference instructions after measurements are final. If release publication cannot be completed, keep the package ready and document that step.

The project is complete when CPU tests/lint pass, data/protocol hashes are frozen, authorized runs finish or stop by the specified rules with diagnosis, final artifacts are auditable, documentation uses measured results accurately, and the selected adapter reloads through `predict`. Project completion does not imply the target was achieved.
