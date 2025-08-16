# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Set
import torch
from torch import nn


class Abstract_ReKV(nn.Module):
    """
    轻量版 ReKV 抽象基类：
    - 允许被无参初始化（兼容 HF 的 super() 构造链）
    - 提供 encode_init_prompt / encode_personalized_pair
    - 仅保留“全局图”一帧；与子类 _get_video_features 配合
    - 提供 clear_cache / get_cache / _get_cache / calc_memory_usage
    """
    def __init__(
        self,
        language_model: nn.Module = None,
        vision_tower: nn.Module = None,
        processor: Any = None,
    ):
        super().__init__()
        self.language_model = language_model
        self.vision_tower = vision_tower
        self.processor = processor

        # KV 缓存（HF past_key_values 的兼容容器）
        self.kv_cache = None
        self.debug = False

    # ---------------- 便捷属性 ----------------
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float16

    def get_input_embeddings(self):
        if self.language_model is None:
            raise RuntimeError("language_model 未初始化，无法获取嵌入层。")
        if hasattr(self.language_model, "get_input_embeddings"):
            return self.language_model.get_input_embeddings()
        raise RuntimeError("language_model 不支持 get_input_embeddings()。")

    # --------------- Prompt helpers ---------------
    def get_choosing_prompt(
        self,
        question: str,
        options: List[str],
        mc: bool = True,
        add_image_token: bool = False,
    ) -> str:
        """
        生成多选/单选题提示词。
        - question: 题干
        - options : 选项列表，如 ["cat", "dog", "bird"]
        - mc      : True 表示单选(single-choice)；False 表示多选(multi-select)
        - add_image_token: 是否在最前面加上 <image> 占位符（默认为 False，
                           因为你的评测是先做个性化编码，然后仅用文本问答）
        返回：一段纯文本 prompt
        """
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        n = len(options)
        if n == 0:
            raise ValueError("options 不能为空")
        if n > len(letters):
            raise ValueError(f"选项过多：{n}，最多支持 {len(letters)} 个")

        lines = []
        if add_image_token:
            lines.append("<image>")
        lines.append("You are a helpful visual question answering assistant.")
        lines.append(f"Question: {question}")
        lines.append("Options:")
        for i, opt in enumerate(options):
            lines.append(f"{letters[i]}. {opt}")

        if mc:
            lines.append("Answer with the single letter only (e.g., 'A').")
        else:
            lines.append("Answer with all correct letters, comma-separated (e.g., 'A,C').")

        return "\n".join(lines)


    # --------------- KV 相关：公共 API ---------------
    def clear_cache(self, keep_global: bool = False):
        """
        清理 KV 缓存。
        keep_global=False: 彻底清空（最常用）
        keep_global=True : 只清局部块，尽量保留全局块（若底层实现了接口）
        """
        if self.kv_cache is None:
            return

        for layer_kv in self.kv_cache:
            try:
                if keep_global:
                    if hasattr(layer_kv, "clear_local"):
                        layer_kv.clear_local()
                    elif hasattr(layer_kv, "reset_local"):
                        layer_kv.reset_local()
                    elif hasattr(layer_kv, "clear_except_global"):
                        layer_kv.clear_except_global()
                else:
                    if hasattr(layer_kv, "clear"):
                        layer_kv.clear()
                    elif hasattr(layer_kv, "reset"):
                        layer_kv.reset()
                    elif hasattr(layer_kv, "clear_all"):
                        layer_kv.clear_all()
            except Exception:
                pass

        if not keep_global:
            self.kv_cache = None

    def get_cache(self):
        return self.kv_cache

    def _get_cache(self):
        return self.kv_cache

    def calc_memory_usage(self, include_model: bool = False) -> int:
        """
        估算内存占用（单位：字节）。默认仅统计 KV 缓存；如需连同模型权重一起统计，设 include_model=True。

        注意：这是按张量 numel()*element_size() 粗略估计，
        与实际显存（分配/碎片/缓存）可能有差异。
        """
        def tensor_bytes(t: torch.Tensor) -> int:
            try:
                return t.numel() * t.element_size()
            except Exception:
                return 0

        seen: Set[int] = set()

        def walk_bytes(obj) -> int:
            oid = id(obj)
            if oid in seen:
                return 0
            seen.add(oid)

            # 张量
            if torch.is_tensor(obj):
                return tensor_bytes(obj)

            # HF 标准 past_key_values: tuple(tuple(Tensor, Tensor), ...)
            if isinstance(obj, (list, tuple)):
                s = 0
                for x in obj:
                    s += walk_bytes(x)
                return s

            # 字典
            if isinstance(obj, dict):
                s = 0
                for v in obj.values():
                    s += walk_bytes(v)
                return s

            # 自定义 KV 管理器对象：遍历 __dict__
            if hasattr(obj, "__dict__"):
                s = 0
                for v in obj.__dict__.values():
                    s += walk_bytes(v)
                return s

            return 0

        total = 0

        # 1) KV 缓存
        if self.kv_cache is not None:
            total += walk_bytes(self.kv_cache)

        # 2) 模型权重（可选）
        if include_model:
            try:
                for p in self.parameters():
                    total += tensor_bytes(p.data)
                for b in self.buffers():
                    total += tensor_bytes(b.data if hasattr(b, "data") else b)
            except Exception:
                pass

        return total

    # --------------- 初始化提示编码（可选） ---------------
    @torch.inference_mode()
    def encode_init_prompt(self):
        """
        假设外部已设置 self.init_prompt_ids (1, S)，用于先行写入 KV。
        没有就跳过。
        """
        if getattr(self, "init_prompt_ids", None) is None:
            if self.debug:
                print("[init_prompt] 无 init_prompt_ids，跳过。")
            return

        if self.language_model is None:
            raise RuntimeError("language_model 未初始化。")

        input_ids = self.init_prompt_ids.to(self.device)
        output = self.language_model(input_ids=input_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values
        if self.debug:
            print(f"[init_prompt] forward with length={input_ids.shape[1]}")

    # --------------- 个性化对（只保留全局图） ---------------
    @torch.inference_mode()
    def encode_personalized_pair(self, pair: Dict[str, Any]):
        """
        pair = {
            "id": <name>,
            "category": <str 或空>,
            "images": [PIL.Image 或 ndarray ...] 或 单张,
            "text": <str 或空>
        }

        Prompt:
            <image>
            This image shows: <id>
            [Category: <category>]
            [Description: <text>]

        仅保留“全局图”：如果拿到 (N, F, 3, H, W) 就裁到 F=1（取第一个裁剪）。
        """
        if self.processor is None:
            raise RuntimeError("processor 未初始化。")
        if self.language_model is None:
            raise RuntimeError("language_model 未初始化。")
        if not hasattr(self, "_get_video_features"):
            raise RuntimeError("模型未实现 _get_video_features()。")

        # 1) 文本 prompt
        id_ = pair["id"]
        category = pair.get("category") or ""
        description = pair.get("text") or ""

        opt_lines = []
        if category:
            opt_lines.append(f"Category: {category}")
        if description:
            opt_lines.append(f"Description: {description}")
        opt_text = ("\n" + "\n".join(opt_lines)) if opt_lines else ""

        prompt = f"<image>\nThis image shows: {id_}{opt_text}"

        # 2) 只保留全局图（多张取第一张）
        imgs = pair.get("images", [])
        if isinstance(imgs, list):
            img0 = imgs[0] if len(imgs) > 0 else None
        else:
            img0 = imgs
        if img0 is None:
            raise RuntimeError(f"pair {id_} 未提供图片。")

        image_inputs = self.processor.image_processor(images=[img0], return_tensors="pt")

        # 有的处理器返回 "pixel_values"，有的返回 "pixel_values_videos"
        pixel_values = image_inputs.get("pixel_values", None)
        if pixel_values is None:
            pixel_values = image_inputs.get("pixel_values_videos", None)
        if pixel_values is None:
            raise RuntimeError("image_processor 未返回 pixel_values。")

        # 对齐到视觉塔 device/dtype，避免 Float/Half 冲突
        vt = getattr(self, "vision_tower", None)
        if vt is None:
            raise RuntimeError("vision_tower 未初始化。")
        try:
            vt_param = next(vt.parameters())
            pixel_values = pixel_values.to(device=vt_param.device, dtype=vt_param.dtype)
        except StopIteration:
            pass

        # 期望 _get_video_features 输入为 (1, F, 3, H, W)
        # 兼容三种情况： (N,3,H,W) / (N,F,3,H,W) / (N,1,F,3,H,W)
        if pixel_values.dim() == 4:
            # (N,3,H,W) -> (N,1,3,H,W)
            pixel_values_5d = pixel_values.unsqueeze(1)
        elif pixel_values.dim() == 5:
            # (N,F,3,H,W) -> 只取全局 F=1
            pixel_values_5d = pixel_values[:, :1, ...]
        elif pixel_values.dim() == 6:
            # (N,1,F,3,H,W) 之类的误加一维 -> 压回 (N,F,3,H,W) 再取 F=1
            pixel_values_5d = pixel_values[:, 0, :1, ...]
        else:
            raise RuntimeError(f"不支持的 pixel_values 形状: {pixel_values.shape}")

        # 3) 视觉特征：输出 (1, 196, D_lang)
        image_features = self._get_video_features(pixel_values_5d)
        lm_param = next(self.language_model.parameters())
        image_features = image_features.to(device=lm_param.device, dtype=lm_param.dtype)

        # 4) 文本 → embedding
        tok = self.processor.tokenizer
        text_inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
        text_inputs = {k: v.to(lm_param.device) for k, v in text_inputs.items()}

        text_emb = self.get_input_embeddings()(text_inputs["input_ids"])  # (1, S, D_lang)
        if text_emb.dtype != lm_param.dtype:
            text_emb = text_emb.to(lm_param.dtype)

        # 5) 用 <image> 占位替换；没有就 [图像, 文本] 拼接
        image_token_id = tok.convert_tokens_to_ids("<image>")
        input_ids_1d = text_inputs["input_ids"][0]  # (S,)
        try:
            if image_token_id is not None:
                image_pos = (input_ids_1d == image_token_id).nonzero(as_tuple=True)[0]
            else:
                image_pos = torch.empty(0, dtype=torch.long, device=lm_param.device)
        except Exception:
            image_pos = torch.empty(0, dtype=torch.long, device=lm_param.device)

        if image_pos.numel() > 0:
            parts: List[torch.Tensor] = []
            last = 0
            for pos in image_pos.tolist():
                if pos > last:
                    parts.append(text_emb[:, last:pos, :])  # 文本片段
                parts.append(image_features)                 # (1, 196, D)
                last = pos + 1
            if last < text_emb.shape[1]:
                parts.append(text_emb[:, last:, :])
            final_embeddings = torch.cat(parts, dim=1).contiguous()  # (1, *, D)
        else:
            final_embeddings = torch.cat([image_features, text_emb], dim=1).contiguous()

        # 6) life-cycle hooks（如果 KV 管理器实现了）
        if self.kv_cache is not None:
            for layer_kv in self.kv_cache:
                if hasattr(layer_kv, "set_current_encoding_pair"):
                    try:
                        layer_kv.set_current_encoding_pair(id_, image_features, text_emb)
                    except Exception:
                        pass

        # 7) 推入 LM，累计 KV
        out = self.language_model(
            inputs_embeds=final_embeddings,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        self.kv_cache = out.past_key_values

        # 8) finalize hook
        if self.kv_cache is not None:
            for layer_kv in self.kv_cache:
                if hasattr(layer_kv, "finalize_current_pair"):
                    try:
                        layer_kv.finalize_current_pair()
                    except Exception:
                        pass

        return self.kv_cache
