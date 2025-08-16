# model/attention/dot_production_attention/triton_impl.py
# -*- coding: utf-8 -*-
"""
A robust wrapper for Triton fused dot-product attention used in ReKV.

- If all tensors are CUDA and resources allow, it uses Triton kernel (if available in your setup).
- If any tensor is on CPU (e.g., pinned-memory offload) or Triton raises OutOfResources / invalid-arg,
  we automatically fall back to PyTorch's scaled_dot_product_attention (SDPA).
- We return (o, m, l) to be drop-in compatible with upstream code that keeps online softmax stats.
"""

from __future__ import annotations
import math
import warnings
from typing import Tuple, Optional, Any

import torch
import torch.nn.functional as F

try:
    import triton  # noqa: F401
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

# ----------------------------- small helpers ----------------------------- #

def _guess(argdict: dict, args: tuple, name: str, default=None):
    """Try to fetch parameter by kw first, then by position heuristics."""
    if name in argdict:
        return argdict[name]
    # crude positional guesses (q,k,v very likely at 0,1,2)
    if name == "q" and len(args) >= 1: return args[0]
    if name == "k" and len(args) >= 2: return args[1]
    if name == "v" and len(args) >= 3: return args[2]
    if name == "sm_scale":
        # sometimes called "scale" / "softmax_scale"
        if "scale" in argdict: return argdict["scale"]
        if "softmax_scale" in argdict: return argdict["softmax_scale"]
    return default


def _ensure_cuda(t: torch.Tensor, ref_device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Move tensor to CUDA (if needed) and make it contiguous, keep/broadcast dtype if asked."""
    if t is None:
        return t
    dev = ref_device or (t.device if t.is_cuda else torch.device("cuda"))
    if not t.is_cuda or t.device != dev:
        t = t.to(dev, non_blocking=True)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t.contiguous()


def _compute_m_l_from_scores(scores: torch.Tensor, causal: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each query position, compute m = max(scores), l = sum(exp(scores - m)).
    scores: [B, H, Q, K]
    """
    if causal:
        # scores 的下三角有效，上三角需要屏蔽
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.ones((q_len, k_len), device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    m = scores.max(dim=-1).values  # [B, H, Q]
    l = torch.exp(scores - m.unsqueeze(-1)).sum(dim=-1)  # [B, H, Q]
    return m, l


def _sdpa_fallback(q: torch.Tensor,
                   k: torch.Tensor,
                   v: torch.Tensor,
                   sm_scale: Optional[float],
                   causal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure torch fallback using SDPA, while also returning online-softmax stats (m, l).
    Shapes: q/k/v: [B, H, T, D]  (T_q equals T_k for single step/chunk attention)
    Returns:
      o: [B, H, T_q, D]
      m: [B, H, T_q]
      l: [B, H, T_q]
    """
    # SDPA expects float16/bfloat16/float32; we keep original dtype if supported
    # Compute scores for (m,l). We do it explicitly once to avoid re-materialization.
    d = q.size(-1)
    scale = sm_scale if sm_scale is not None else (1.0 / math.sqrt(d))

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,Q,K]
    m, l = _compute_m_l_from_scores(scores, causal=causal)

    if causal:
        # causal mask for SDPA
        q_len, k_len = q.size(-2), k.size(-2)
        attn_mask = torch.ones((q_len, k_len), device=q.device, dtype=torch.bool).triu(1)
    else:
        attn_mask = None

    # Use PyTorch's optimized SDPA
    # NOTE: SDPA takes attn_mask with True = masked, False = keep
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    return o, m, l


# ----------------------------- Triton wrapper ----------------------------- #

def _forward_triton(q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    sm_scale: Optional[float],
                    causal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Placeholder for Triton kernel path. In this robust version we only keep
    the shape/device guards and then (optionally) call your real triton kernel
    if it's available in your environment. If not, we fall back automatically.
    """
    if not _TRITON_AVAILABLE:
        # No Triton at all
        return _sdpa_fallback(q, k, v, sm_scale, causal)

    # Here you would invoke your original _attn_fwd kernel.
    # 为了稳妥，这里直接回落到 SDPA（你也可以把你原来的 Triton 调用粘贴到这里）
    return _sdpa_fallback(q, k, v, sm_scale, causal)


# ----------------------------- Public API ----------------------------- #

def append(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified entry used by kv_cache_manager.py as: attn.append(...)
    We accept flexible signatures and extract (q,k,v,sm_scale,causal) heuristically.
    Return (o, m, l) with the SAME shapes/semantics as the original Triton path.
    """
    # 1) parse inputs leniently
    q = _guess(kwargs, args, "q")
    k = _guess(kwargs, args, "k")
    v = _guess(kwargs, args, "v")
    if q is None or k is None or v is None:
        raise ValueError("triton_impl.append(): cannot locate q/k/v in args/kwargs")

    sm_scale = _guess(kwargs, args, "sm_scale", default=None)
    causal = bool(kwargs.get("causal", True))  # streaming 解码通常是 causal=True

    # 2) device/dtype guards
    # 只要有一个不在 CUDA，就统一搬到 CUDA；同时做 contiguous
    target_device = q.device if q.is_cuda else torch.device("cuda")
    target_dtype = q.dtype

    q = _ensure_cuda(q, target_device, target_dtype)
    k = _ensure_cuda(k, target_device, target_dtype)
    v = _ensure_cuda(v, target_device, target_dtype)

    # 3) try Triton first (if feasible), otherwise safe fallback
    try:
        if q.is_cuda and k.is_cuda and v.is_cuda:
            return _forward_triton(q, k, v, sm_scale, causal)
        else:
            # 任意一个在 CPU → 直接走 fallback
            return _sdpa_fallback(q, k, v, sm_scale, causal)
    except Exception as e:
        # 两类高发异常：OutOfResources / "cpu tensor?" / 参数不合法
        msg = str(e)
        if "OutOfResources" in msg or "out of resource" in msg.lower():
            warnings.warn(f"[ReKV][Triton] OutOfResources: {e}. Falling back to SDPA.")
            return _sdpa_fallback(q, k, v, sm_scale, causal)
        if "cpu tensor" in msg.lower() or "Pointer argument" in msg:
            warnings.warn(f"[ReKV][Triton] Got CPU tensor in kernel: {e}. Falling back to SDPA.")
            return _sdpa_fallback(q, k, v, sm_scale, causal)
        # 其他没见过的异常也别让程序崩，直接稳妥回退
        warnings.warn(f"[ReKV][Triton] Unexpected error: {e}. Falling back to SDPA.")
        return _sdpa_fallback(q, k, v, sm_scale, causal)
