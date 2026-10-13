"""Pinned model loading, CUDA checks, and adapter discovery."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from rl_countdown.config import ProjectConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def cuda_diagnostic() -> dict[str, Any]:
    """Report CUDA availability using PyTorch and the NVIDIA management CLI."""
    report: dict[str, Any] = {"ok": False, "checks": {}}
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        report["checks"]["torch"] = torch.__version__
        report["checks"]["cuda_available"] = cuda
        report["checks"]["device_count"] = int(torch.cuda.device_count())
        if cuda:
            device = torch.cuda.get_device_properties(0)
            torch.empty((128, 128), dtype=torch.bfloat16, device="cuda").matmul(
                torch.empty((128, 128), dtype=torch.bfloat16, device="cuda")
            )
            torch.cuda.synchronize()
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            report["device"] = torch.cuda.get_device_name(0)
            report["total_memory_gib"] = round(device.total_memory / (1024**3), 3)
            report["free_memory_gib"] = round(free_bytes / (1024**3), 3)
            report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            report["cuda_runtime_version"] = torch.version.cuda
            if not report["bf16_supported"]:
                raise RuntimeError("The selected training precision requires BF16 support.")
            if total_bytes < (7 * 1024**3):
                raise RuntimeError(f"The target requires a GPU with at least 7 GiB of VRAM; detected {total_bytes}.")
            report["ok"] = True
        else:
            report["error"] = "PyTorch cannot see an available NVIDIA CUDA device."
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        candidates = (
            Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
            Path(r"C:\Program Files (x86)\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
            Path(r"C:\Windows\System32\nvidia-smi.exe"),
        )
        nvidia_smi = next((str(path) for path in candidates if path.is_file()), None)
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            report["nvidia_smi"] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            report["nvidia_smi_error"] = f"{type(error).__name__}: {error}"
    return report


def require_cuda() -> dict[str, Any]:
    report = cuda_diagnostic()
    if not report["ok"]:
        raise RuntimeError(f"CUDA preflight failed: {json.dumps(report, sort_keys=True)}")
    return report


def load_base_model(config: ProjectConfig) -> tuple[Any, Any]:
    """Download/load the single, immutable model checkpoint and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    project = config.project
    model_id = str(project["model_id"])
    revision = str(project["model_revision"])
    if model_id != MODEL_ID or revision != MODEL_REVISION:
        raise ValueError("The model family and immutable revision are part of the frozen experiment protocol.")
    if not torch.cuda.is_available():
        raise RuntimeError("Call doctor and enable the RTX 4070 before loading this training model.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not all(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("All original model parameters must remain frozen before the LoRA adapter is attached.")
    model.config.use_cache = False
    return model, tokenizer


def discover_adapter_registry(root: Path) -> dict[str, dict[str, Any]]:
    registry_path = root / "runs" / "experiment_registry.json"
    if not registry_path.exists():
        return {"schema_version": 1, "experiments": {}}
    content = json.loads(registry_path.read_text(encoding="utf-8"))
    if content.get("schema_version") != 1 or not isinstance(content.get("experiments"), dict):
        raise ValueError(f"Unsupported experiment registry: {registry_path}")
    return content


def save_adapter_registry(root: Path, registry: dict[str, dict[str, Any]]) -> Path:
    path = root / "runs" / "experiment_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path

