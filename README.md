# Countdown RLVR

An auditable reinforcement-learning post-training experiment for a small instruction model. The project tests whether **direct GRPO with a shaped, exactly verifiable reward** improves a model's ability to solve Countdown arithmetic puzzles in one response.

This is a deliberately narrow experiment. It can support a claim about arithmetic puzzle solving under the stated protocol; it cannot establish that the model gained general reasoning ability. The project compares RL with the untouched model and with a separately trained supervised control.

> **Results status: pending.** Training and evaluation have not produced measured results yet. The table below must be filled only from committed evaluation artifacts. Do not cite a planned target as an achieved result.

<!-- MEASURED_RESULTS_START -->
| Model / run | ID greedy accuracy | OOD greedy accuracy | Status |
|---|---:|---:|---|
| Public instruction checkpoint | pending | pending | Pending measurement |
| Direct GRPO, seed 42 | pending | pending | Pending measurement |
| Direct GRPO, seed 43 | pending | pending | Pending measurement |
| Direct GRPO, seed 44 | pending | pending | Pending measurement |
| Validity-reward ablation, seed 42 | pending | pending | Pending measurement |
| SFT control, seed 42 | pending | pending | Pending measurement |
<!-- MEASURED_RESULTS_END -->

## Experiment at a glance

| Component | Design |
|---|---|
| Starting model | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Adaptation | LoRA, with original model weights frozen |
| Main method | Direct GRPO using TRL and the Dr. GRPO loss; no project SFT before RL |
| Task | Use each supplied integer exactly once to form a target with `+`, `-`, `*`, `/`, and parentheses |
| Main reward | `1.0` for an exact solution, `0.2` for a valid but incorrect expression, and `0` for invalid output |
| Data | Deterministically generated, exact-verified train, validation, ID test, and OOD test splits |
| Controls | Original checkpoint, validity-only reward ablation, and independent SFT adapter |
| Compute limit | At most 72 accumulated GPU-hours, with no paid compute or services |

The base checkpoint is already instruction-tuned. “Direct GRPO” means this repository does not add its own supervised fine-tuning stage before reinforcement learning. The independent SFT control uses training solutions as labels so its results can be compared transparently with RL.

## Task and verification

For example, given `[2, 3, 7]` and target `20`, `(7+3)*2` is a solution. Each response must contain exactly one expression. A restricted parser checks the expression and evaluates it with exact rational arithmetic. It rejects extra numbers, missing or repeated inputs, unsupported syntax, explanations, Markdown, and division by zero. It never executes model-generated code.

The dataset generator uses a deterministic seed and an exact dynamic-programming solver to produce a valid witness for each problem. A number multiset is assigned to at most one ID partition, so permutations or alternate targets cannot leak across train, validation, and test. GRPO receives the problem only; it does not receive the witness. SFT reads witnesses from train only. See [the frozen protocol](docs/protocol.md) for the full data, training, and evaluation specification.

## Reproduce

The intended training environment is Python 3.11 on Windows with a CUDA-enabled PyTorch 2.8.0 build targeting CUDA 12.8. The repository pins the other runtime dependencies in `uv.lock`.

```powershell
uv sync --group dev --extra gpu
uv run rl-countdown doctor
uv run rl-countdown prepare-data
```

`doctor` checks CUDA, BF16 support, the active GPU, and available disk space. Do not start model downloads or training if the GPU check fails. The CPU-only test suite and lint checks run in CI without downloading model weights:

```powershell
uv sync --group dev
uv run pytest
uv run ruff check .
```

Run the experiment stages with the shared frozen configuration. `configs/experiments.yaml` records the experiment matrix, seeds, and reward variants:

```powershell
uv run rl-countdown baseline --config configs/default.yaml
uv run rl-countdown smoke --config configs/default.yaml
uv run rl-countdown pilot --config configs/default.yaml
uv run rl-countdown train-grpo --config configs/default.yaml
uv run rl-countdown train-sft --config configs/default.yaml
uv run rl-countdown evaluate --config configs/default.yaml
uv run rl-countdown report --config configs/default.yaml
```

The engineering smoke run uses only 16 balanced validation problems to verify save, resume, and adapter loading quickly. The 128-row validation subset is reserved for pilot and training-checkpoint comparisons; the final report evaluates the full 512-row validation split.

Training commands accept `--resume`; a resumed run must have matching model, data, and configuration hashes. Each run uses its own output directory, and a new run must not overwrite an existing one. For local inference, point `--adapter` at a selected trained adapter:

```powershell
uv run rl-countdown predict --config configs/default.yaml --nums 2 3 7 --target 20 --adapter runs/grpo_correct_seed_42/best
```

The command names and supported options are the public CLI contract. Read the committed run manifests and `configs/` files for exact experiment configuration and artifact paths.

## Results and acceptance

The primary metric is greedy one-response accuracy on the frozen ID test set. The OOD test uses numbers from a disjoint, larger range and is reported separately; improvement on OOD is not a success requirement. Report validity, accuracy conditional on validity, scores by puzzle arity, output length, truncation, latency, training time, and peak GPU memory alongside the primary metric.

The pre-registered target is a mean improvement of at least 10 percentage points on ID test, a paired 95% bootstrap interval above zero, and no main seed below the original checkpoint. These are **targets**, not results. An honest negative result completes the project if the protocol, runs, and evidence are reported accurately. The full selection, stopping, and uncertainty rules are in [the protocol](docs/protocol.md).

## Limitations

- Countdown has an exact classical solver. This project studies whether a small instruction model can learn the task through verifiable-reward post-training; it does not claim the LLM is the best solver for this task.
- A synthetic benchmark is narrow and may reward patterns specific to its generator. ID performance does not show broad reasoning ability.
- OOD changes the number range only. It is a limited generalization check, not a comprehensive robustness evaluation.
- The shaped reward gives valid-but-incorrect expressions 0.2 and exact solutions 1.0; the validity-only ablation isolates the contribution of exact correctness. Reward-group variation is logged because GRPO cannot learn from uniform groups.
- SFT has reference expressions and is not compute- or information-equivalent to GRPO. Report its label access and cost explicitly.
- Results depend on the stated prompt, parser, decoding settings, data hashes, and hardware. Do not compare numbers produced under a different protocol as if they were the same experiment.

## Interview guide and artifacts

The [Spanish interview guide](docs/interview.md) gives a short project explanation and prepares answers about reward design, leakage, controls, and limitations. `results/` is reserved for selected evidence such as metrics, predictions, manifests, and figures. Training checkpoints and model weights belong in ignored local run directories, not in Git. The seed-42 adapter is intended for a `v0.1.0` GitHub release with a checksum and loading instructions after results have been measured.
