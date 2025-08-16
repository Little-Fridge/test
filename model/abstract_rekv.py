import torch
from logzero import logger


class Abstract_ReKV:
    """
    抽象基类（mixin）：
    - 负责通用的 KV 管理、init prompt 编码、个性化 pair 编码（图像 + 文本）
    - 视觉特征提取 _get_video_features() 由子类实现（返回形如 (1, F*196, D)）
    - 具体的 QA / VQA / REC 逻辑由子类实现
    """

    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        self.processor = processor
        self.n_frame_tokens = int(n_frame_tokens)
        self.init_prompt_ids = init_prompt_ids  # 支持 None / str / list[int] / (L,) or (1,L) Tensor
        self.n_local = int(n_local)
        self.topk = int(topk)
        self.chunk_size = int(chunk_size)

    # -------------------- device / dtype / embeddings --------------------
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float16 if torch.cuda.is_available() else torch.float32

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # -------------------- cache utils --------------------
    def clear_cache(self):
        self.kv_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # -------------------- init-prompt 编码（稳健版） --------------------
    @torch.inference_mode()
    def encode_init_prompt(self):
        """
        将 self.init_prompt_ids 规范化为 (1, L) 的 LongTensor。
        - None / 空 → 设为 (1,0) 空张量，n_init=0，直接返回（不做前向）
        - str       → tokenizer.encode → (1, L)
        - list/tuple→ as_tensor → (1, L)
        - Tensor    → (L,) → (1, L)；(1, L) 保持
        最终仅当 L>0 时才执行 language_model 前向并写入 kv_cache。
        """
        dev = self.device

        def _set_empty_and_return():
            self.n_init = 0
            self.init_prompt_ids = torch.empty((1, 0), dtype=torch.long, device=dev)
            logger.debug("[init_prompt] empty (skip forward)")
            return self.kv_cache

        x = self.init_prompt_ids

        # 1) None / 空：优雅短路
        if x is None:
            return _set_empty_and_return()
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return _set_empty_and_return()
        if torch.is_tensor(x) and x.numel() == 0:
            return _set_empty_and_return()
        if isinstance(x, str) and len(x.strip()) == 0:
            return _set_empty_and_return()

        # 2) 规范成 2D (1, L)
        if isinstance(x, str):
            tok = getattr(getattr(self, "processor", None), "tokenizer", None)
            if tok is None:
                return _set_empty_and_return()
            ids = tok.encode(x, add_special_tokens=False)
            ids = torch.as_tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)  # (1, L)
        elif torch.is_tensor(x):
            ids = x.to(device=dev, dtype=torch.long)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)  # (1, L)
            elif ids.dim() == 2:
                pass
            else:
                raise ValueError(f"[init_prompt] tensor dim must be 1 or 2, got {ids.shape}")
        else:
            # 视作可迭代的 token id 列表
            ids = torch.as_tensor(x, dtype=torch.long, device=dev)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)  # (1, L)
            elif ids.dim() != 2:
                ids = ids.view(1, -1)

        L = int(ids.size(-1))
        self.init_prompt_ids = ids
        self.n_init = L

        if L == 0:
            return _set_empty_and_return()

        # 3) 正式前向（input_ids 必须为 2D）
        logger.debug(f"[init_prompt] forward with length={L}")
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values
        return self.kv_cache

    # -------------------- 视觉特征提取（由子类实现） --------------------
    def _get_video_features(self, pixel_values_videos):
        """
        子类需要实现：
        输入：pixel_values_videos, 形如 (B=1, F, 3, H, W)
        输出：video_features, 形如 (1, F*196, D)
        """
        raise NotImplementedError

    # -------------------- 将视频分块编码进 KV（通用） --------------------
    def _encode_video_chunk(self, video_chunk):
        """
        video_chunk: (Nv, H, W, 3) numpy/array-like
        使用 processor.video_processor → pixel_values_videos: (1, Nv, 3, H, W)
        然后 _get_video_features → (1, Nv*196, D) → 送入 LLM 以累计 KV。
        """
        batch = self.processor.video_processor(videos=[video_chunk], return_tensors="pt")
        pixel_values_videos = batch.get("pixel_values", batch)
        pixel_values_videos = pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)

        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
        assert self.n_local >= video_features.shape[1], \
            f"n_local: {self.n_local}, video_features: {video_features.shape[1]}"

        output = self.language_model(
            inputs_embeds=video_features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True
        )
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):
        """
        video: (Nv, H, W, 3)
        分块送入，逐块累计 KV。
        """
        num_frames = int(video.shape[0])
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in range(num_chunks):
            st = chunk_idx * encode_chunk_size
            ed = st + encode_chunk_size
            chunk_video = video[st:ed]
            self._encode_video_chunk(chunk_video)

        if num_frames % encode_chunk_size != 0:
            self._encode_video_chunk(video[num_chunks * encode_chunk_size:])

        return self.kv_cache

    # -------------------- 个性化 pair 编码（图像 + 文本，训练对齐版） --------------------
    @torch.inference_mode()
    def encode_personalized_pair(self, pair):
        """
        pair = {
            "id": name,
            "category": category,
            "images": [PIL.Image or ndarray ...] or single image,
            "text": text
        }

        生成 prompt：
            <image>
            Name: <id>
            [Category: <category>]
            [Description: <text>]
        并将 <image> 替换为图像特征；若无 <image> token，则按 [image_features, text_embeddings] 拼接。
        """
        # ---------- 1) 拼文本 ----------
        id_ = pair["id"]
        category = pair.get("category", "")
        description = pair.get("text", "")

        optional = ""
        if category:
            optional += f"\nCategory: {category}"
        if description:
            optional += f"\nDescription: {description}"

        if isinstance(pair.get("images", []), list) and len(pair["images"]) > 1:
            prompt = f"<image>\nThese {len(pair['images'])} images show the personalized content: {id_}{optional}"
        else:
            prompt = f"<image>\nThis image shows: {id_}{optional}"

        # ---------- 2) 图像 → feature ----------
        images = pair["images"] if isinstance(pair.get("images", []), list) else [pair["images"]]
        image_inputs = self.processor.image_processor(images=images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self.device, self.dtype)  # (N, 3, H, W)

        # 期望下游 _get_video_features 输入为 (1, F, 3, H, W)
        if pixel_values.dim() == 4:  # (N, 3, H, W)
            pixel_values = pixel_values.unsqueeze(0)  # (1, N, 3, H, W)

        image_features = self._get_video_features(pixel_values)  # (1, N*196, D)
        if image_features.dtype != self.dtype:
            image_features = image_features.to(self.dtype)

        # ---------- 3) 文本 → embeddings ----------
        text_inputs = self.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_embeddings = self.get_input_embeddings()(text_inputs["input_ids"])  # (1, S, D)
        if text_embeddings.dtype != self.dtype:
            text_embeddings = text_embeddings.to(self.dtype)

        # ---------- 4) 将 <image> token 替换为图像特征 ----------
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        input_ids_1d = text_inputs["input_ids"][0]  # (S,)
        try:
            image_positions = (input_ids_1d == image_token_id).nonzero(as_tuple=True)[0] \
                if image_token_id is not None else torch.empty(0, dtype=torch.long, device=self.device)
        except Exception:
            image_positions = torch.empty(0, dtype=torch.long, device=self.device)

        if image_positions.numel() > 0:
            final_embeddings_list = []
            last_pos = 0
            for img_pos in image_positions.tolist():
                if img_pos > last_pos:
                    final_embeddings_list.append(text_embeddings[0, last_pos:img_pos])  # 文本片段
                final_embeddings_list.append(image_features[0])  # (N*196, D)
                last_pos = img_pos + 1
            if last_pos < text_embeddings.shape[1]:
                final_embeddings_list.append(text_embeddings[0, last_pos:])
            final_embeddings = torch.cat(final_embeddings_list, dim=0).unsqueeze(0)  # (1, *, D)
        else:
            # 若 tokenizer 不含 <image>，保持与训练一致：图像特征在前
            final_embeddings = torch.cat([image_features, text_embeddings], dim=1)

        # ---------- 5) pair lifecycle：让 KV 知道当前是在写哪个 concept ----------
        if self.kv_cache is not None:
            for layer_kv in self.kv_cache:
                if hasattr(layer_kv, "set_current_encoding_pair"):
                    layer_kv.set_current_encoding_pair(id_, image_features, text_embeddings)

        # ---------- 6) 推入模型，累计 KV ----------
        output = self.language_model(
            inputs_embeds=final_embeddings,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True
        )
        self.kv_cache = output.past_key_values

        # 写回 blocks → concept 的映射
        if self.kv_cache is not None:
            for layer_kv in self.kv_cache:
                if hasattr(layer_kv, "finalize_current_pair"):
                    layer_kv.finalize_current_pair()

        return self.kv_cache

    # -------------------- 预留接口：由子类实现 --------------------
    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        raise NotImplementedError

    # -------------------- KV 内存估算 --------------------
    def calc_memory_usage(self):
        if self.kv_cache is None:
            return 0
        if isinstance(self.kv_cache, (list, tuple)) and len(self.kv_cache) == 0:
            return 0
        try:
            n_layers = len(self.kv_cache)
            memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
            return memory
        except Exception:
            # 若实现方没有 calculate_cpu_memory，可自定义估算或返回 0
            return 0
