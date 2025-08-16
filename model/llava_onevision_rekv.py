# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple, Union
import inspect
import warnings
import torch
from torch import nn

from transformers import AutoProcessor
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionForConditionalGeneration,
)

from .abstract_rekv import Abstract_ReKV


def _first_param_device_dtype(m: Optional[nn.Module]) -> Tuple[torch.device, torch.dtype]:
    if m is not None:
        try:
            p = next(m.parameters())
            return p.device, p.dtype
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float32


class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    """
    ReKV + LLaVA-OneVision（切片→分块→检索）
    - 支持 (1,F,G,3,H,W) 与 (1,F,3,H,W)；默认每切片取 196 patch
    - projector 映射后输出 (1, N*T, D_lang)，交由 Abstract_ReKV 切分写入 KV
    """

    def __init__(self, config):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)

        # 缺省参数（外部可通过 load_model 写回覆盖）
        if not hasattr(self, "n_frame_tokens"):
            self.n_frame_tokens = 196
        if not hasattr(self, "block_size"):
            self.block_size = 196       # 方便“一切片≈一块”的直觉
        if not hasattr(self, "only_global"):
            self.only_global = False    # False=用多切片；True=只用全局图

        self.kv_cache = None

    # 安全 tie_weights
    def tie_weights(self):
        lm = getattr(self, "language_model", None)
        if lm is not None and hasattr(lm, "tie_weights"):
            try:
                lm.tie_weights()
            except Exception as e:
                warnings.warn(f"[tie_weights] 跳过异常：{e}")

    # ---- 多切片特征提取（关键） ----
    @torch.inference_mode()
    def _get_video_features(self, pixel_values_5d: torch.Tensor) -> torch.Tensor:
        x = pixel_values_5d
        if x.dim() == 6:
            # (1, F, G, 3, H, W)
            B, F, G, C, H, W = x.shape
            if B != 1 or C != 3:
                raise RuntimeError(f"[vt] 期望 (1, F, G, 3, H, W)，得到 {x.shape}")
            if self.only_global:
                x = x[:, :, :1]  # 只保留全局图
                B, F, G, C, H, W = x.shape
            N = F * G
            pv = x.reshape(N, C, H, W)
        elif x.dim() == 5:
            # (1, F, 3, H, W)
            B, F, C, H, W = x.shape
            if B != 1 or C != 3:
                raise RuntimeError(f"[vt] 期望 (1, F, 3, H, W)，得到 {x.shape}")
            pv, N = x.reshape(F, C, H, W), F
        else:
            raise RuntimeError(f"[vt] 非法像素维度：{x.shape}，需要 5D/6D")

        vt: nn.Module = getattr(self, "vision_tower", None)
        if vt is None:
            raise RuntimeError("vision_tower 未初始化。")

        vt_dev, vt_dtype = _first_param_device_dtype(vt)
        pv = pv.to(device=vt_dev, dtype=vt_dtype)

        vout = vt(pv, output_hidden_states=True)
        tokens = (
            vout.last_hidden_state if hasattr(vout, "last_hidden_state")
            else vout[0] if isinstance(vout, (tuple, list)) else vout
        )
        if tokens.dim() != 3:
            raise RuntimeError(f"[vt] 非法 tokens 形状：{tokens.shape}")

        L = tokens.shape[1]
        T = int(getattr(self, "n_frame_tokens", 196))

        if L == T:
            patch = tokens
        elif L >= T + 1:
            patch = tokens[:, 1:1 + T]  # 丢 CLS
        elif L > T:
            patch = tokens[:, :T]
        else:
            pad = torch.zeros(tokens.size(0), T - L, tokens.size(-1),
                              device=tokens.device, dtype=tokens.dtype)
            patch = torch.cat([tokens, pad], dim=1)  # (N,T,Dv)

        projector = getattr(self, "multi_modal_projector", None) or getattr(self, "visual_projector", None)
        if projector is None:
            raise RuntimeError("未找到 projector（multi_modal_projector / visual_projector）。")

        feats = projector(patch)               # (N,T,D_lang)
        feats = feats.reshape(1, N * T, -1)    # (1, N*T, D_lang)
        return feats

    # ---- 选项 prompt（兼容 list/dict） ----
    def get_choosing_prompt(self, question: str, options: Union[List[str], Dict[str, str]], mc: bool = True) -> str:
        labels = ["A", "B", "C", "D"]
        if isinstance(options, dict):
            ordered = []
            for i, lab in enumerate(labels):
                v = options.get(lab, options.get(str(i), options.get(i)))
                if v is not None:
                    ordered.append(f"{lab}. {v}")
        else:
            ordered = [f"{labels[i]}. {options[i]}" for i in range(min(len(options), 4))]

        opt_text = "\n".join(ordered)
        return f"Question: {question}\nOptions:\n{opt_text}\nAnswer with the single letter only (e.g., 'A')."

    # ---- VQA（对齐 dtype/device，过滤不被 forward 接收的键） ----
    @torch.inference_mode()
    def visual_question_answering(
        self,
        image: "PIL.Image.Image",
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        **gen_kwargs,
    ) -> str:
        if self.processor is None:
            raise RuntimeError("processor 尚未绑定，请在 load_model 中确保已创建并回填。")

        # 1) 构造 chat 模板（带 <image>）
        user_text = f"<image>\n{prompt}"
        messages = [{"role": "user", "content": user_text}]
        tok = self.processor.tokenizer
        chat_str = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # 2) 处理输入张量，分别对齐到 lm/vision 的 device&dtype
        inputs = self.processor(text=chat_str, images=image, return_tensors="pt")

        lm = getattr(self, "language_model", None)
        vt = getattr(self, "vision_tower", None)

        def _dev_dtype(m):
            try:
                p = next(m.parameters())
                return p.device, p.dtype
            except Exception:
                return (torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float32)

        lm_dev, _ = _dev_dtype(lm if lm is not None else self)
        vt_dev, vt_dtype = _dev_dtype(vt if vt is not None else self)

        model_inputs = {}
        for k, v in inputs.items():
            if not torch.is_tensor(v):
                continue
            if k in ("pixel_values", "pixel_values_videos"):
                model_inputs[k] = v.to(device=vt_dev, dtype=vt_dtype)
            else:
                model_inputs[k] = v.to(device=lm_dev)

        # 3) 生成参数
        gen_params = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )
        if temperature is not None:
            gen_params["temperature"] = float(temperature)
        # Qwen2 一些权重没有 pad_token_id，这里兜底一下
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            gen_params["pad_token_id"] = tok.eos_token_id

        gen_params.update(gen_kwargs)

        # 4) 生成 & 只解码“新增 token”
        out = self.generate(**model_inputs, **gen_params)             # (1, prompt_len + new_len)
        seq = out[0]
        in_len = model_inputs["input_ids"].shape[1] if "input_ids" in model_inputs else 0
        gen_only = seq[in_len:]                                       # 只取新增部分
        text = tok.decode(gen_only, skip_special_tokens=True).strip()
        return text


    # ---- 直接读取“检索命中”的辅助（不依赖 reset_retrieval） ----
    @torch.inference_mode()
    def get_debug_retrieved_indices(self) -> Optional[List]:
        if not hasattr(self, "kv_cache") or self.kv_cache is None:
            return None
        hits = []
        for layer in self.kv_cache:
            # 常见字段名都试一下
            cand = None
            for attr in (
                "retrieved_block_indices",
                "last_retrieved",
                "debug_last_indices",
                "retrieved_indices",
            ):
                if hasattr(layer, attr) and getattr(layer, attr) is not None:
                    cand = getattr(layer, attr)
                    break
            hits.append(cand)
        # 全是 None 就返回 None
        if all(h is None for h in hits):
            return None
        return hits


# ------------------ 统一加载入口 ------------------
def load_model(
    model_path: str,
    torch_dtype: Any = "auto",
    device_map: Any = "auto",
    **kwargs,
):
    rekv_keys = {
        "n_init", "n_local", "fattn", "block_size", "topk", "chunk_size",
        "max_cached_block", "exc_block_size", "pin_memory", "n_frame_tokens",
        "init_exc", "only_global"
    }
    rekv_opts: Dict[str, Any] = {}
    hf_opts: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in rekv_keys:
            rekv_opts[k] = v
        else:
            hf_opts[k] = v

    model = LlavaOneVision_ReKV.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **hf_opts,
    )
    processor = AutoProcessor.from_pretrained(model_path)

    model.processor = processor
    if not hasattr(model, "language_model") or model.language_model is None:
        raise RuntimeError("language_model 未加载到模型中。")
    if not hasattr(model, "vision_tower") or model.vision_tower is None:
        raise RuntimeError("vision_tower 未加载到模型中。")

    for k, v in rekv_opts.items():
        setattr(model, k, v)

    try:
        model.tie_weights()
    except Exception as e:
        warnings.warn(f"[load_model] 调用 tie_weights 异常：{e}")

    return model, processor
