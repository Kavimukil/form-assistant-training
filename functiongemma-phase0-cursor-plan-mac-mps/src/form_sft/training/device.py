from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DevicePlan:
    device: str
    torch_dtype: torch.dtype
    bf16: bool
    fp16: bool


def select_device(requested: str = "auto") -> DevicePlan:
    """Select a portable Phase-0 training backend.

    Priority for auto: CUDA -> Apple MPS -> CPU.
    CUDA may use BF16/FP16 through TrainingArguments. MPS intentionally uses
    float32 model weights and disables CUDA-oriented bf16/fp16 trainer flags
    for the conservative Phase-0 baseline.
    """
    requested = requested.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {requested}")

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        bf16 = bool(torch.cuda.is_bf16_supported())
        return DevicePlan("cuda", torch.bfloat16 if bf16 else torch.float16, bf16, not bf16)

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
        return DevicePlan("mps", torch.float32, False, False)

    return DevicePlan("cpu", torch.float32, False, False)
