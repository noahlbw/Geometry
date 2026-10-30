"""Execution-only optimizations for the frozen Geov2 Anchored path.

This module deliberately contains no model rule.  It only changes storage
and numerical-kernel choices around the already frozen Anchored equations:

    F^W = Y0 + beta (Pi_hat - Pi) C_sem
    Z   = L + g0 H (W - L)

The helpers are opt-in so old reports and exact compatibility runs keep their
original path.  The fast deployment uses the same observations and writer,
but materializes class logits tile by tile and stores a Cholesky factor for
the structural operator instead of an explicit H matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AnchoredInfraConfig:
    """Safe execution knobs; none changes the Anchored algebra."""

    solver_dtype: str = "float32"
    stream_logits: bool = True
    fast_math: bool = True
    weight_dtype: str = "none"
    wide_policy: str | None = None

    def validate(self) -> None:
        if self.solver_dtype not in ("float32", "float64"):
            raise ValueError("solver_dtype must be float32 or float64")
        if self.weight_dtype not in ("none", "bfloat16"):
            raise ValueError("weight_dtype must be none or bfloat16")


def configure_fast_math(enabled: bool = True) -> dict[str, object]:
    """Enable tensor-core kernels without changing any model operation.

    TF32 is used only for float32 matrix multiplies on Ampere+ GPUs.  All
    semantic vectors and the structural writer still leave the model in the
    dtypes selected by their callers.
    """
    previous = {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
    }
    if enabled and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return previous


def _cast_linear_storage(module: nn.Module, dtype: torch.dtype) -> dict[str, int]:
    """Cast frozen linear/conv weights only, leaving normalization in FP32.

    DINO's autocast already executes these layers in BF16 on A800.  Storing
    their frozen parameters in BF16 removes the duplicate FP32 residency,
    while keeping LayerNorm, RoPE and other numerically sensitive buffers in
    their original dtype.
    """
    changed = {"parameters": 0, "bytes_before": 0, "bytes_after": 0}
    seen: set[int] = set()
    for submodule in module.modules():
        if not isinstance(submodule, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            continue
        for parameter in (submodule.weight, submodule.bias):
            if parameter is None or id(parameter) in seen or not parameter.is_floating_point():
                continue
            seen.add(id(parameter))
            before = parameter.numel() * parameter.element_size()
            if parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)
                changed["parameters"] += 1
            changed["bytes_before"] += before
            changed["bytes_after"] += parameter.numel() * parameter.element_size()
    return changed


def cast_shared_visual_weights(segmenter, weight_dtype: str = "none") -> dict[str, object]:
    """Optionally reduce the resident frozen visual model footprint.

    Only the visual backbone/head is touched.  Text caches, local/wide
    formulas, alias aggregation, and structural tensors remain unchanged.
    Calling the helper twice is idempotent.
    """
    if weight_dtype not in ("none", "bfloat16"):
        raise ValueError("weight_dtype must be none or bfloat16")
    if weight_dtype == "none":
        return {"weight_dtype": "none", "parameters": 0, "bytes_before": 0, "bytes_after": 0}
    visual = segmenter.model.visual_model
    stats = _cast_linear_storage(visual, torch.bfloat16)
    stats["weight_dtype"] = "bfloat16"
    return stats


def cast_anchored_visual_weights(model, weight_dtype: str = "none") -> dict[str, object]:
    """Cast the frozen visual modules used by both Anchored views.

    ``FrozenGeometry`` keeps a Geometry visual head and an independent wide
    QuerySlot head.  The two heads must remain separate, while their frozen
    backbone may already be shared by :class:`OptimizedWideReadout`.  This
    helper therefore walks both visual modules, deduplicates shared
    parameters, and casts only Linear/Conv storage.  LayerNorm, RoPE and all
    relation/slot tensors stay in their original dtype.  It is an opt-in
    storage/precision path; the default ``weight_dtype='none'`` is a no-op.
    """
    if weight_dtype not in ("none", "bfloat16"):
        raise ValueError("weight_dtype must be none or bfloat16")
    if weight_dtype == "none":
        return {
            "weight_dtype": "none",
            "visual_modules": 0,
            "parameters": 0,
            "bytes_before": 0,
            "bytes_after": 0,
        }
    modules = []
    for branch in (getattr(model, "geometry", None), getattr(model, "vip", None)):
        backbone = getattr(branch, "backbone", None)
        visual = getattr(getattr(backbone, "model", None), "visual_model", None)
        if visual is not None and all(id(visual) != id(previous) for previous in modules):
            modules.append(visual)
    if not modules:
        raise ValueError("Anchored model has no visual modules to cast.")
    total = {"weight_dtype": "bfloat16", "visual_modules": len(modules),
             "parameters": 0, "bytes_before": 0, "bytes_after": 0}
    seen: set[int] = set()
    for visual in modules:
        stats = _cast_linear_storage_unique(visual, torch.bfloat16, seen)
        for key in ("parameters", "bytes_before", "bytes_after"):
            total[key] += stats[key]
    return total


def _cast_linear_storage_unique(module: nn.Module, dtype: torch.dtype,
                                seen: set[int]) -> dict[str, int]:
    """Shared-parameter variant of :func:`_cast_linear_storage`."""
    changed = {"parameters": 0, "bytes_before": 0, "bytes_after": 0}
    for submodule in module.modules():
        if not isinstance(submodule, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            continue
        for parameter in (submodule.weight, submodule.bias):
            if parameter is None or id(parameter) in seen or not parameter.is_floating_point():
                continue
            seen.add(id(parameter))
            before = parameter.numel() * parameter.element_size()
            if parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)
                changed["parameters"] += 1
            changed["bytes_before"] += before
            changed["bytes_after"] += parameter.numel() * parameter.element_size()
    return changed


def factor_solver(dtype: str = "float32"):
    """Return the factorized reconstruction closure used by Anchored.

    ``factor_reconstruction`` returns ``(R, valid)`` with
    ``R R^T = I + A^T A``.  StaticReader applies
    ``H D = D - solve(R R^T, D)``, which is algebraically the same
    ``H=(I+A^T A)^{-1}A^T A`` action as the historical explicit operator.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    from .geov2_wide_optimized import factor_reconstruction

    torch_dtype = getattr(torch, dtype)

    def solve(relation, valid):
        return factor_reconstruction(relation, valid, torch_dtype)

    return solve


def prepare_streamed_source(source, *, reader, profile, bank_name, wide_scorer,
                            local_scorer, cache=None):
    """Collapse high-dimensional Anchored observations to logits.

    The returned source has the same local positions/operators and the same
    precomputed wide canvas as the ordinary reader.  It drops each tile's
    ``[N,D]`` semantic feature and the wide feature list after evaluating the
    exact frozen scorers once.  StaticReader recognizes the two private fields
    and therefore applies the unchanged writer and blending order.

    This is intentionally restricted to retained/union-max background rules;
    residual ontology readers need their extra wide evidence and stay on the
    compatibility path.
    """
    if profile.background_rule not in ("retained", "union_max"):
        return source
    if not source.get("wide") or not source.get("local"):
        return source
    if isinstance(source["wide"], dict):
        return source
    cache = {} if cache is None else cache
    query = reader.queries[bank_name]
    bank = reader.banks[bank_name]
    # This is exactly the call StaticReader would make.  Its result is kept
    # as the compact CxHxW canvas; all 1024xD wide features can be released.
    wide = wide_scorer(source, query, profile.tau, profile.tem)
    if isinstance(wide, dict):
        return source
    members = reader.members.get(bank_name)
    text = reader.text[bank_name]
    prepared_local = []
    for tile in source["local"]:
        feature = tile["features"][profile.strength]
        alias = feature.float() @ text.T
        local = local_scorer.uniform_reference(alias)
        if profile.background_rule == "union_max" and reader.background is not None:
            local[:, reader.background] = alias[:, members].amax(-1)
        prepared_local.append(dict(tile, local_logits=local.detach()))
    compact = dict(source, local=prepared_local, wide=[],
                   _precomputed_wide=wide.detach())
    return compact


def describe_config(config: AnchoredInfraConfig) -> dict[str, object]:
    config.validate()
    return {
        "solver_dtype": config.solver_dtype,
        "stream_logits": bool(config.stream_logits),
        "fast_math": bool(config.fast_math),
        "weight_dtype": config.weight_dtype,
        "wide_policy_override": config.wide_policy,
    }
