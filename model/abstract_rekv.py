import torch
from logzero import logger


class Abstract_ReKV:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size

    def clear_cache(self):
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def _get_video_features(self, pixel_values_videos):
        pass

    def _encode_video_chunk(self, video_chunk):
        pixel_values_videos = self.processor.video_processor(video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):  # video: (Nv, H, W, 3)
        # encode chunk by chunk
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            self._encode_video_chunk(remaining_video)
        
        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')


    @torch.inference_mode()
    def encode_personalized_pair(self, pair):
        '''
        pair = {
            "id": name,
            "category": category,
            "images": [images],
            "text": text
        }
        '''
        '''encode a personalized pair of image and text with prompt:
            <image>
            Name: <id>
            Category: <category>
            Description: <text>
        '''
        # generate the personalization prompt
        category = pair.get('category', '')
        description = pair.get('text', '')
        id_ = pair['id']

        # 拼接可选部分
        optional = ""
        if category:
            optional += f"\nCategory: {category}"
        if description:
            optional += f"\nDescription: {description}"

        if isinstance(pair['images'], list) and len(pair['images']) > 1:
            prompt = f"<image>\nThese {len(pair['images'])} images show the personalized content: {id_}{optional}"
        else:
            prompt = f"<image>\nThis image shows: {id_}{optional}"
            
        # 分别处理图像和文本
        # 1. 处理图像：提取图像特征
        if isinstance(pair['images'], list):
            # 如果是多张图片，需要处理成类似视频帧的格式
            images = pair['images']
        else:
            images = [pair['images']]
        
        # 使用image_processor单独处理图像，获取pixel_values
        image_inputs = self.processor.image_processor(
            images=images,
            return_tensors="pt"
        )
        pixel_values = image_inputs['pixel_values'].to(self.device, self.dtype)
        
        # 将图像格式化为类似视频的格式: (1, num_images, 3, H, W)
        if len(pixel_values.shape) == 4:  # (num_images, 3, H, W)
            pixel_values = pixel_values.unsqueeze(0)  # (1, num_images, 3, H, W)
        
        # 使用_get_video_features方法处理图像
        image_features = self._get_video_features(pixel_values)  # (1, num_images*196, D)
        
        # 2. 处理文本：转换为embeddings
        text_inputs = self.processor.tokenizer(
            prompt, 
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        
        text_embeddings = self.get_input_embeddings()(text_inputs.input_ids)  # (1, seq_len, D)
        
        # 3. 合并图像特征和文本embeddings
        # 需要找到<image>标记的位置并替换为图像特征
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids('<image>')
        input_ids = text_inputs.input_ids[0]  # (seq_len,)
        
        # 找到<image>标记的位置
        image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        
        if len(image_positions) > 0:
            # 构建最终的embeddings序列
            final_embeddings_list = []
            last_pos = 0
            
            for img_pos in image_positions:
                # 添加<image>之前的文本embeddings
                if img_pos > last_pos:
                    final_embeddings_list.append(text_embeddings[0, last_pos:img_pos])
                
                # 添加图像特征
                final_embeddings_list.append(image_features[0])  # (num_images*196, D)
                
                last_pos = img_pos + 1
            
            # 添加最后剩余的文本embeddings
            if last_pos < text_embeddings.shape[1]:
                final_embeddings_list.append(text_embeddings[0, last_pos:])
            
            # 拼接所有embeddings
            final_embeddings = torch.cat(final_embeddings_list, dim=0).unsqueeze(0)  # (1, total_len, D)
        else:
            # 如果没有找到<image>标记，直接拼接
            final_embeddings = torch.cat([image_features, text_embeddings], dim=1)

        # 检查并padding到块大小的倍数（对于ReKV的offloading机制）
        seq_len = final_embeddings.shape[1]
        block_size = self.n_frame_tokens  # 196
        if seq_len % block_size != 0:
            # 需要padding到下一个块大小的倍数
            target_len = ((seq_len // block_size) + 1) * block_size
            padding_len = target_len - seq_len
            
            # 创建padding token embeddings (使用tokenizer的pad_token或者零向量)
            if hasattr(self.processor.tokenizer, 'pad_token_id') and self.processor.tokenizer.pad_token_id is not None:
                pad_token_id = self.processor.tokenizer.pad_token_id
                pad_embeddings = self.get_input_embeddings()(torch.tensor([[pad_token_id] * padding_len], device=self.device))
            else:
                # 如果没有pad_token，使用零向量padding
                embed_dim = final_embeddings.shape[-1]
                pad_embeddings = torch.zeros(1, padding_len, embed_dim, device=final_embeddings.device, dtype=final_embeddings.dtype)
            
            # 添加padding
            final_embeddings = torch.cat([final_embeddings, pad_embeddings], dim=1)

        # 4. 传递给language_model，使用inputs_embeds而不是原始inputs
        output = self.language_model(
            inputs_embeds=final_embeddings,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True
        )

        # update kv-cache
        self.kv_cache = output.past_key_values

        return self.kv_cache
    
    
    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory
