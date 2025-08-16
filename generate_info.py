#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_info.py

为 yollava-data/train/<concept>/ 下的每个概念自动生成 info 文本描述，
供 ReKV 各种测试脚本直接 import 使用。

Usage as a module:
    from generate_info import generate_all_info
    info_map = generate_all_info(
        train_root="yollava-data/train",
        client=client,
        model_name="gpt-4o-mini"
    )

Usage as a CLI:
    python generate_info.py \
        --train_root ./yollava-data/train \
        --output infos.json \
        --api_key YOUR_API_KEY \
        --base_url https://api.gptplus5.com/v1 \
        --model gpt-4o-mini
"""
import argparse
import json
import logging
import random
import time
import base64
import io
from pathlib import Path
from typing import Dict, List

from PIL import Image
from openai import OpenAI

# 日志打印级别
logging.basicConfig(level=logging.INFO)


def encode_image_to_base64(image_path: str) -> str:
    """
    将图片转换为 base64 编码；若宽度 > 2048，则等比缩放至 2048 宽。
    并且在保存为 JPEG 前强制转为 RGB，避免 RGBA 模式保存失败。
    """
    with Image.open(image_path) as img:
        # 强制转换为 RGB 模式
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        # 超宽图片等比缩放
        if w > 2048:
            scale = 2048 / w
            img = img.resize((2048, int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        # 保存为 JPEG（此时一定是 RGB 模式）
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def chat_with_images_gpt(
    client: OpenAI,
    prompt: str,
    image_paths: List[str],
    model_name: str,
    max_retries: int = 5
) -> str:
    """
    调用 GPT-4 Vision（通过 OpenAI client）处理文本+多张图片，返回模型回复。
    带指数退避重试机制。
    """
    retryable = ("rate_limit", "timeout", "connection_error",
                 "server_error", "500", "502", "503", "504")
    for attempt in range(max_retries):
        try:
            # 构造消息：文本 + data-url 图片
            content = [{"type": "text", "text": prompt}]
            for path in image_paths:
                b64 = encode_image_to_base64(path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=512
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if any(tok in msg for tok in retryable) and attempt < max_retries - 1:
                delay = min(5 * (2 ** attempt), 60)
                jitter = delay * 0.1
                actual = delay + random.uniform(-jitter, jitter)
                logging.warning(f"第 {attempt+1} 次请求失败：{e}，{actual:.1f}s 后重试")
                time.sleep(actual)
                continue
            logging.error(f"请求终止，错误：{e}")
            raise


def generate_all_info(
    train_root: str,
    client: OpenAI,
    model_name: str
) -> Dict[str, str]:
    """
    扫描 train_root/{concept} 目录，为每个概念调用 GPT-4 Vision 生成英文 info 描述。
    """
    root = Path(train_root)
    if not root.is_dir():
        raise FileNotFoundError(f"未找到 train_root：{root}")
    info_map: Dict[str, str] = {}

    for concept_dir in sorted(root.iterdir()):
        if not concept_dir.is_dir():
            continue
        # 收集所有图片路径
        imgs = [str(p) for p in sorted(concept_dir.glob("*.*"))
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not imgs:
            logging.warning(f"{concept_dir.name} 下无图片，跳过")
            continue

        # 英文 prompt：突出视觉特征、典型场景和偏好
        prompt = (
            f'You will see several reference images of the same concept named "{concept_dir.name}".\n'
            f'Write EXACTLY ONE English sentence (≤18 words) that starts with: "The {concept_dir.name} is ".\n'
            "Pack dense, observable cues as short phrases separated by commas, in this order:\n"
            "colors → shape/silhouette → textures/materials → distinctive parts/markings → pose → 1–3‑word setting.\n"
            "Rules: no talk about images or design, no hedges (typically/often/may), no opinions/emotions,\n"
            "use only nouns/adjectives after 'is', avoid 'and', avoid repeated words, keep it concrete.\n"
            "Return ONLY the single sentence.\n"
            "Style examples:\n"
            'The husky toy is white‑gray, plush texture, triangular ears, blue eyes, curled tail, lying on a sofa.\n'
            'The red can opener is compact, curved handle, metal jaws, gear wheel, on a kitchen counter.'
        )
        
        logging.info(f"正在为 “{concept_dir.name}” 生成 info ({len(imgs)} 张图)…")
        info = chat_with_images_gpt(
            client=client,
            prompt=prompt,
            image_paths=imgs,
            model_name=model_name
        ).strip().replace("\n", " ")
        logging.info(f"  → {info}\n")
        info_map[concept_dir.name] = info

    return info_map


def main():
    parser = argparse.ArgumentParser(
        description="为 yollava-data/train 下每个概念生成英文 info 描述"
    )
    parser.add_argument(
        "--train_root", "-t",
        type=str,
        default="./yollava-data/train",
        help="yollava-data/train 根目录"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="infos.json",
        help="输出 JSON 路径"
    )
    parser.add_argument(
        "--api_key", "-k",
        type=str,
        default='sk-YhR1FycY6wzVIwSaAbC3FaE8571141A29aCb14E4A27dAc8b',
        help="OpenAI API Key"
    )
    parser.add_argument(
        "--base_url", "-u",
        type=str,
        default="https://api.gptplus5.com/v1",
        help="API 基础 URL"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="GPT Vision 模型名称"
    )
    args = parser.parse_args()

    # 初始化 OpenAI Client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    # 生成并保存 infos
    info_map = generate_all_info(
        train_root=args.train_root,
        client=client,
        model_name=args.model
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(info_map, f, ensure_ascii=False, indent=2)
    logging.info(f"[完成] 已生成 {len(info_map)} 条描述，保存在 {args.output}")


if __name__ == "__main__":
    main()
