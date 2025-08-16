#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rekv_personalization_text.py

用于测试 yollava-text-qa 数据集的个性化测试，
从已有的 infos.json 直接读取 info 文本，不再重复生成。
"""

import argparse
import json
import re
import logging
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict  # 新增：用于按概念统计

# 把项目根目录加入 Python 路径，确保能 import model
sys.path.append(str(Path(__file__).resolve().parent))

from model import llava_onevision_rekv

logging.basicConfig(level=logging.INFO)

# 全局：KV 检索信息捕获
last_retrieved_indices = None

def setup_retrieval_capture(model):
    """打补丁捕获 KV-Cache 检索到的块索引"""
    global last_retrieved_indices
    last_retrieved_indices = None
    if not getattr(model, 'kv_cache', None):
        return
    for layer in model.kv_cache:
        if hasattr(layer, 'reset_retrieval'):
            orig = layer.reset_retrieval
            def wrap(orig_fn, kv_layer):
                def wrapped():
                    global last_retrieved_indices
                    idxs = getattr(kv_layer, 'retrieved_block_indices', None)
                    if isinstance(idxs, list):
                        last_retrieved_indices = [i.copy() if isinstance(i, list) else i for i in idxs]
                    else:
                        last_retrieved_indices = idxs
                    return orig_fn()
                return wrapped
            layer.reset_retrieval = wrap(orig, layer)

def get_last_retrieved_indices():
    return last_retrieved_indices

def test_personalization(personalize_set, qa, model_name):
    """主测试流程：注入 KV，跑 text-qa，多选接口"""
    model, processor = llava_onevision_rekv.load_model(
        model_path=f"model_zoo/{model_name}",
        n_local=7500,
        topk=8,
        chunk_size=1
    )
    model.clear_cache()
    model.encode_init_prompt()
    setup_retrieval_capture(model)

    print("===============编码个性化数据集===============")
    for name_, content in personalize_set.items():
        print(f"encoding {name_} ...")
        pair = {
            "id": name_,
            "category": content['category'],
            "images": content["images"],
            "text": content['info']
        }
        model.encode_personalized_pair(pair)
        print(f"  KV-Cache 使用内存: {model.calc_memory_usage()/(1024**3):.2f} GB")
    torch.cuda.synchronize()

    # 兜底重编码无块的 pair
    layer0 = model.kv_cache[0] if model.kv_cache else None
    if layer0:
        redo = [pid for pid,info in layer0.pairs.items() if len(info.get('blocks',[]))==0]
        if redo:
            print(f"[RE-ENCODE] 以下 pairs 无 blocks，重编码: {redo}")
            for pid in redo:
                content = personalize_set[pid]
                model.encode_personalized_pair({
                    "id": pid,
                    "category": content['category'],
                    "images": content['images'],
                    "text": content['info']
                })
            torch.cuda.synchronize()

    print("\n=== VQA 多选测试 ===")
    correct, total = 0, 0
    # 新增：按概念累计
    per_concept = defaultdict(lambda: {"correct": 0, "total": 0})

    for qa_ in qa:
        q_text = qa_['question'].replace('sks', qa_['name'])
        prompt = model.get_choosing_prompt(q_text, qa_['options'], mc=True)
        print(f"\n[Q] {q_text}")
        print(f"prompt: {prompt}")
        print("  检索到的块索引:", get_last_retrieved_indices())
        answer = model.question_answering({"question": q_text, "prompt": prompt}, max_new_tokens=128)
        raw_choice = answer.strip().split()[0] if answer.strip().split() else ''
        m = re.search(r'([A-Z])', raw_choice)
        choice = m.group(1) if m else raw_choice
        pred_text = qa_['options'].get(choice, "")
        print(f"[Pred] {choice}) {pred_text}    [GT] {qa_['correct_answer']}")

        # 全局累计
        if choice == qa_['correct_answer']:
            correct += 1
            per_concept[qa_['name']]["correct"] += 1
        per_concept[qa_['name']]["total"] += 1
        total += 1

        print(f"   Acc: {correct}/{total} = {correct/total:.3f}")
        print("  检索到的块索引:", get_last_retrieved_indices())

    print(f"\n=== Final Accuracy: {correct}/{total} = {correct/total:.3f} ===")

    # 新增：输出每个 concept 的准确率
    print("\n=== 每个概念的准确率 ===")
    # 为了保持稳定顺序，按名称排序输出
    for name in sorted(per_concept.keys()):
        c = per_concept[name]["correct"]
        t = per_concept[name]["total"]
        acc = (c / t) if t else 0.0
        print(f"{name}: {acc:.3f} ({c}/{t})")

def main():
    p = argparse.ArgumentParser(description="ReKV Text-QA 个性化测试（读取 infos.json 注入 info）")
    p.add_argument("--data_root", "-d", type=str, default="./yollava-data",
                   help="数据根目录，包含 train/ 和 test/")
    p.add_argument("--model_name", "-m", type=str, default="LLaVA/llava-onevision-qwen2-7b-ov-hf")
    args = p.parse_args()

    train_root = os.path.join(args.data_root, "train")
    test_json = os.path.join(args.data_root, "text_qa.json")
    ds = json.load(open(test_json, encoding="utf-8"))

    # 1) 构造 personalize_set，不含 info
    personalize_set = {}
    for name, items in ds.items():
        imgs = []
        for fn in sorted(os.listdir(os.path.join(train_root, name))):
            imgs.append(Image.open(os.path.join(train_root, name, fn)).convert("RGB"))
        personalize_set[name] = {
            "category": name,
            "images": imgs,
            "info": None
        }

    # 2) 直接读取已生成的 infos.json
    infos_path = "/home/ubuntu/ken/ReKV/infos.json"
    info_map = json.load(open(infos_path, encoding="utf-8"))
    logging.info(f"[INFO] 从 {infos_path} 读取到 {len(info_map)} 条描述")

    # 3) 注入 info 到 personalize_set
    for name in personalize_set:
        personalize_set[name]['info'] = info_map.get(name, "")

    # 4) 构造 QA 列表
    qa = []
    for name, items in ds.items():
        for _, qa_item in items.items():
            qa.append({
                "name": name,
                "question": qa_item['question'],
                "options": qa_item['option'],
                "correct_answer": qa_item['correct_answer']
            })

    # 5) 运行测试
    test_personalization(personalize_set, qa, args.model_name)

if __name__ == "__main__":
    main()
