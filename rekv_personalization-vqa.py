#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用于测试 yollava-visual-qa 数据集的个性化测试（仅检索 infos.json 的文字信息）

修复/改动：
1) 移除 visual_question_answering() 不支持的 temperature 参数。
2) 不再把含 <|im_start|>/<|im_end|> 的字符串传入模型；使用纯文本 MC 提示。
3) 仅注入/检索 infos.json 文本：pair 中 images=[]，减少无用 KV blocks 与显存。
4) 健壮解析模型输出中的选项字母。
5) 修复检索信息捕获的 monkey-patch。
"""

import re
import json
import sys
import os
from PIL import Image

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import llava_onevision_rekv

# ============ 检索信息捕获 ============
last_retrieved_indices = None

def setup_retrieval_capture(model):
    """设置检索信息捕获（修复了 isinstance 的用法）"""
    global last_retrieved_indices
    last_retrieved_indices = None

    if not hasattr(model, 'kv_cache') or not model.kv_cache:
        print("警告: kv_cache未创建，无法设置检索信息捕获")
        return

    for i, layer_kv in enumerate(model.kv_cache):
        if not hasattr(layer_kv, 'reset_retrieval'):
            continue
        original_reset = layer_kv.reset_retrieval

        def make_wrapped_reset(original_func, kv_layer):
            def wrapped_reset(*args, **kwargs):
                global last_retrieved_indices
                try:
                    if hasattr(kv_layer, 'retrieved_block_indices') and kv_layer.retrieved_block_indices is not None:
                        r = kv_layer.retrieved_block_indices
                        if isinstance(r, list):
                            last_retrieved_indices = [x.copy() if isinstance(x, list) else x for x in r]
                        else:
                            last_retrieved_indices = r
                    else:
                        last_retrieved_indices = None
                except Exception:
                    last_retrieved_indices = None
                return original_func(*args, **kwargs)
            return wrapped_reset

        layer_kv.reset_retrieval = make_wrapped_reset(original_reset, layer_kv)

def get_last_retrieved_indices():
    return last_retrieved_indices

# ============ 纯文本多选提示 & 解析 ============
def build_mc_prompt(question: str, options: dict) -> str:
    """
    构造不含任何特殊 chat token 的纯文本提示。
    """
    lines = [question.strip(), "Options:"]
    for k in sorted(options.keys()):
        lines.append(f"{k}. {options[k]}")
    # 强约束，只输出单个大写字母
    lines.append("Answer with ONLY a single capital letter (e.g., A). Do not output anything else.")
    return "\n".join(lines)

def extract_choice_letter(text: str, valid_keys=("A","B","C","D","E")):
    """
    稳健抽取选项字母。允许 A / A) / A. / 'Best option: A' 等。
    """
    if not text:
        return None
    s = str(text).strip()

    # 单独的大写字母
    m = re.search(r'\b([A-Z])\b', s)
    if m and m.group(1) in valid_keys:
        return m.group(1)

    # A) / A. / A:
    m = re.search(r'\b([A-Z])(?=[\)\.\:\s])', s)
    if m and m.group(1) in valid_keys:
        return m.group(1)

    # Best option: A
    m = re.search(r'[Bb]est\s*[Oo]ption\s*[:：]\s*([A-Z])', s)
    if m and m.group(1) in valid_keys:
        return m.group(1)

    return None

# ============ 主流程 ============
def test_personalization(personalize_set, qa, model_name="LLaVA/llava-onevision-qwen2-7b-ov-hf"):
    model_path = f'model_zoo/{model_name}'

    model, processor = llava_onevision_rekv.load_model(
        model_path=model_path,
        n_local=3920,    # 本地窗口大小
        topk=8,          # 检索块数
        chunk_size=1     # 块大小
    )

    model.clear_cache()
    model.encode_init_prompt()

    # 仅检索文字描述（infos.json）
    if hasattr(model, "kv_cache") and model.kv_cache:
        for layer_kv in model.kv_cache:
            if hasattr(layer_kv, "set_retrieval_mode"):
                layer_kv.set_retrieval_mode("text")
    else:
        print("警告：模型未暴露 kv_cache，无法设置 retrieval_mode")

    # 捕获检索信息
    setup_retrieval_capture(model)

    # 编码个性化数据集 —— 只注入文本，images 置空，避免无谓的 KV 占用
    print(f"===============编码个性化数据集===============")
    print(f"Personalization pairs num: {len(personalize_set)}")
    for name_, content in personalize_set.items():
        print(f"encoding {name_}...")
        pair = {
            "id": name_,
            "category": content.get('category', None),
            "images": [],                      # 关键：text-only 模式，清空图片
            "text": content.get('info', None)
        }
        model.encode_personalized_pair(pair)
        memory_usage = model.calc_memory_usage() / (1024**3)
        print(f"KV-Cache memory usage: {memory_usage:.2f} GB")

    # 测试 VQA（纯文本 MC 提示；不要传任何含特殊 token 的字符串）
    correct, total = 0, 0
    for qa_ in qa:
        q_text = qa_['question'].replace('<sks>', qa_['name'])
        prompt_txt = build_mc_prompt(q_text, qa_['options'])
        print("\n===============VQA测试===============")
        print("question:", q_text)
        print("prompt:\n", prompt_txt)

        # 同时传入 question 原始问题 & prompt（完整 MC 提示）
        out = model.visual_question_answering(
            qa_['image'],
            {"question": q_text, "prompt": prompt_txt},
            max_new_tokens=4   # 只需 1 个字母，给 4 足够
        )

        raw = out if isinstance(out, str) else (out[0] if out else "")
        print("模型原始输出:", raw)

        pred_choice = extract_choice_letter(raw, valid_keys=tuple(qa_['options'].keys()))
        print("解析到的选项字母:", pred_choice)
        print("标准答案:", qa_['correct_answer'])

        if pred_choice == qa_['correct_answer']:
            correct += 1
        total += 1
        acc = correct / total if total else 0.0
        print(f"累计准确率: {correct}/{total} = {acc:.4f}")

        retrieved_indices = get_last_retrieved_indices()
        if retrieved_indices is not None:
            print(f"检索到的块索引: {retrieved_indices}")
        else:
            print("未检索到相关信息")

    return acc


if __name__ == "__main__":
    json_path = './yollava-data/yollava-visual-qa.json'
    ds = json.loads(open(json_path, encoding="utf-8").read())

    personalize_set = {}
    qa = []
    for name in ds:
        personalize_set[name] = {
            "category": None,
            "info": None,
            "images": []  # 将不会被使用
        }

        # 题目
        for j in ds[name]:
            qa.append({
                "name": name,
                "question": ds[name][j]['question'],
                "image": Image.open(j).convert("RGB"),
                "options": ds[name][j]['options'],
                "correct_answer": ds[name][j]['correct_answer']
            })

    # 从 infos.json 注入 info（仅文本）
    infos_path = "infos.json"
    try:
        with open(infos_path, "r", encoding="utf-8") as f:
            info_map = json.load(f)
        print(f"[INFO] 加载 infos.json 成功，共 {len(info_map)} 条。")
    except Exception as e:
        print(f"[WARN] 读取 {infos_path} 失败：{e}")
        info_map = {}

    for name in personalize_set:
        if name in info_map and isinstance(info_map[name], str):
            personalize_set[name]["info"] = info_map[name]

    print("个性化数据集已加载，包含以下项目:")
    for k, v in personalize_set.items():
        print(f"name: {k}, category: {v['category']}, info: {v['info']}, images num: {len(v['images'])}")
    print("开始个性化测试（仅文本检索）...")

    test_personalization(
        personalize_set=personalize_set,
        qa=qa,
        model_name="LLaVA/llava-onevision-qwen2-7b-ov-hf"
    )
