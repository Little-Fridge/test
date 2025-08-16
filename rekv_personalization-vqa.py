#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用于测试 yollava-visual-qa 数据集的个性化测试（切片→分块→检索）
"""

import json
import os
import sys
from typing import Any, Dict

import torch
from PIL import Image

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import llava_onevision_rekv  # noqa: E402

# ---------------- 全局：用于保存检索信息（兼容旧方式） ----------------
last_retrieved_indices = None


def setup_retrieval_capture(model):
    """旧方式：wrap reset_retrieval，在 reset 前抓取命中块。"""
    global last_retrieved_indices
    last_retrieved_indices = None

    if not hasattr(model, "kv_cache") or model.kv_cache is None:
        print("警告: kv_cache未创建，无法设置检索信息捕获")
        return

    for i, layer_kv in enumerate(model.kv_cache):
        if hasattr(layer_kv, "reset_retrieval"):
            original_reset = layer_kv.reset_retrieval

            def make_wrapped_reset(original_func, kv_layer):
                def wrapped_reset():
                    global last_retrieved_indices
                    grab = None
                    for attr in (
                        "retrieved_block_indices",
                        "last_retrieved",
                        "debug_last_indices",
                        "retrieved_indices",
                    ):
                        if hasattr(kv_layer, attr) and getattr(kv_layer, attr) is not None:
                            grab = getattr(kv_layer, attr)
                            break
                    last_retrieved_indices = grab
                    return original_func()

                return wrapped_reset

            layer_kv.reset_retrieval = make_wrapped_reset(original_reset, layer_kv)


def get_last_retrieved_indices(model=None):
    """优先用新接口读；没有再退回旧缓存。"""
    if model is not None and hasattr(model, "get_debug_retrieved_indices"):
        hits = model.get_debug_retrieved_indices()
        if hits is not None:
            return hits
    return last_retrieved_indices


def _tensor_bytes_any(x: Any) -> int:
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes_any(t) for t in x)
    if isinstance(x, dict):
        return sum(_tensor_bytes_any(v) for v in x.values())
    # 尝试对象属性（如果有 __dict__）
    try:
        return sum(_tensor_bytes_any(v) for v in vars(x).values())
    except Exception:
        return 0


def get_kv_cache_gb(model) -> float:
    total = 0
    if hasattr(model, "kv_cache") and model.kv_cache is not None:
        for layer in model.kv_cache:
            total += _tensor_bytes_any(layer)
    return total / (1024 ** 3)


def test_personalization(personalize_set: Dict, qa: list):
    model_path = "model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf"

    model, processor = llava_onevision_rekv.load_model(
        model_path=model_path,
        n_local=4096,     # 本地窗口
        topk=8,           # 每层检索块数
        chunk_size=1,     # 1块=1切片×196token
        block_size=196,   # 保持和上面一致
        only_global=False # 使用多切片（例如21）
    )

    # ---- 启用检索（非常关键） ----
    # 你的 Abstract_ReKV 里一般以这些属性开关检索；这里显式打开以防默认关闭
    setattr(model, "fattn", True)            # 打开检索注意力（你的项目里一般这样命名）
    if getattr(model, "topk", 0) <= 0:
        setattr(model, "topk", 8)
    if not hasattr(model, "n_local"):
        setattr(model, "n_local", 4096)
    if not hasattr(model, "chunk_size"):
        setattr(model, "chunk_size", 1)

    # 清空并创建 KV 结构
    if hasattr(model, "clear_cache"):
        model.clear_cache()
    if hasattr(model, "_get_cache"):
        model._get_cache()

    # 可选：初始化提示
    if hasattr(model, "encode_init_prompt"):
        model.encode_init_prompt()

    # 旧式抓取挂钩（保底）
    setup_retrieval_capture(model)

    # 编码个性化数据集
    print("===============编码个性化数据集===============")
    print(f"Personalization pairs num: {len(personalize_set)}")
    for name_, content in personalize_set.items():
        print(f"encoding {name_}...")
        pair = {
            "id": name_,
            "category": content["category"],
            "images": content["images"],  # List[PIL.Image.Image]
            "text": content["info"],
        }
        model.encode_personalized_pair(pair)

        memory_usage = get_kv_cache_gb(model)
        print(f"KV-Cache memory usage: {memory_usage:.2f} GB")

    # VQA
    correct_num = 0
    corrent_num = 0
    for qa_ in qa:
        q_text = qa_["question"].replace("<sks>", qa_["name"])
        prompt = model.get_choosing_prompt(q_text, qa_["options"], mc=True)

        print("===============VQA测试===============")
        print(f"question: {q_text}")
        print(f"prompt: {prompt}")

        # 生成
        ans_text = model.visual_question_answering(
            qa_["image"], prompt, max_new_tokens=64, do_sample=False, temperature=None
        ).strip()

        # 只取第一个大写字母当作选项
        pred = ""
        for ch in ans_text:
            if ch.upper() in ("A", "B", "C", "D"):
                pred = ch.upper()
                break

        print(f"回答: {pred}  原始: {ans_text}")
        print(f"答案: {qa_['correct_answer']}")
        if pred == str(qa_["correct_answer"]).upper():
            correct_num += 1
        corrent_num += 1
        print(f"correct_num: {correct_num}, corrent_num: {corrent_num}, accuracy: {correct_num/corrent_num:.3f}")

        # 读取检索信息（新接口优先，旧缓存兜底）
        hits = get_last_retrieved_indices(model)
        if hits is not None:
            print(f"检索到的块索引: {hits}")
        else:
            print("未检索到相关信息")

    return correct_num, corrent_num


if __name__ == "__main__":
    json_path = "./yollava-data/yollava-visual-qa.json"
    ds = json.loads(open(json_path, "r", encoding="utf-8").read())

    personalize_set = {}
    qa = []
    for name in ds:
        personalize_set[name] = {"category": None, "info": None, "images": []}
        train_dir = f"./yollava-data/train/{name}"
        for filename in os.listdir(train_dir):
            image_path = f"{train_dir}/{filename}"
            personalize_set[name]["images"].append(Image.open(image_path).convert("RGB"))

        for j in ds[name]:
            qa.append(
                {
                    "name": name,
                    "question": ds[name][j]["question"],
                    "image": Image.open(j).convert("RGB"),
                    "options": ds[name][j]["options"],
                    "correct_answer": ds[name][j]["correct_answer"],
                }
            )

    # 从 infos.json 注入 text
    infos_path = "infos.json"
    try:
        with open(infos_path, "r", encoding="utf-8") as f:
            info_map = json.load(f)
        print(f"[INFO] 加载 infos.json 成功，共 {len(info_map)} 条。")
    except Exception as e:
        print(f"[WARN] 读取 {infos_path} 失败：{e}")
        info_map = {}

    print("个性化数据集已加载，包含以下项目:")
    for k, v in personalize_set.items():
        if k in info_map and isinstance(info_map[k], str):
            personalize_set[k]["info"] = info_map[k]
        print(f"name: {k}, category: {v['category']}, info: {personalize_set[k]['info']}, images num: {len(v['images'])}")

    print("图片已经加载，开始个性化测试...")

    test_personalization(
        personalize_set=personalize_set,
        qa=qa,
    )
