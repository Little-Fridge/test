#!/usr/bin/env python3
"""
yollava-data Recognition 任务（带进度打印）：
1. 使用 train/<concept>/ 注入个性化（KV-Cache）。
2. 使用 test/<concept>/ 做识别：
   问题固定 "Is <sks> in this photo?" 选项 Yes/No。
3. 输出每张图片预测 + 最终 Positive / Negative / Weighted Accuracy
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import json
import re  # 新增：稳健解析 A/B

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import llava_onevision_rekv  # noqa

last_retrieved_indices = None


def setup_retrieval_capture(model):
    """捕获检索到的块索引（可选调试）"""
    global last_retrieved_indices
    last_retrieved_indices = None
    if not hasattr(model, 'kv_cache') or model.kv_cache is None:
        return
    for layer_kv in model.kv_cache:
        if hasattr(layer_kv, 'reset_retrieval'):
            original_reset = layer_kv.reset_retrieval

            def make_wrapped_reset(original_func, kv_layer):
                def wrapped_reset():
                    global last_retrieved_indices
                    idxs = getattr(kv_layer, 'retrieved_block_indices', None)
                    if isinstance(idxs, list):
                        last_retrieved_indices = [i.copy() if isinstance(i, list) else i for i in idxs]
                    else:
                        last_retrieved_indices = idxs
                    return original_func()
                return wrapped_reset

            layer_kv.reset_retrieval = make_wrapped_reset(original_reset, layer_kv)


def run_inference(model, image, question):
    """
    只接受开头是 'A' 或 'B' 的输出；否则一律判 B。
    不再用 'yes'/'no' 子串兜底，避免把模型口癖当答案。
    """
    import re

    prompt = model.get_choosing_prompt(
        question + "\nChoose one: A or B. Reply with a single letter only.",
        {"A": "Yes", "B": "No"},
        mc=True
    )

    # 允许 2~3 个 token，让模型把 'A' 或 'B' 吐完整
    ans = model.visual_question_answering(
        image,
        {"question": question, "prompt": prompt},
        max_new_tokens=3
    )

    if not ans:
        return 'B'

    s = ans.strip()
    # 只看开头第一个非空字符是不是 A/B
    m = re.match(r"^\s*([AB])\b", s)
    if m:
        return m.group(1)

    # 开头不是 A/B，一律判 B（更保守，有利于负样本）
    return "B"



if __name__ == "__main__":
    print(">>> Script started", flush=True)

    # 1. 个性化集合：train 目录
    personalize_set = {}
    train_root = './yollava-data/train'
    train_concepts = [c for c in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, c))]
    print(f">>> Building personalize_set from train ({len(train_concepts)} concepts)...", flush=True)
    for idx, name in enumerate(train_concepts, 1):
        sub = os.path.join(train_root, name)
        imgs = []
        for fn in os.listdir(sub):
            p = os.path.join(sub, fn)
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        personalize_set[name] = {
            'category': None,
            'info': None,    # 后续从 infos.json 注入
            'images': imgs
        }
        if idx % 5 == 0 or idx == len(train_concepts):
            print(f"    Loaded {idx}/{len(train_concepts)} concepts", flush=True)

    # --- 新增：从 infos.json 加载所有 info 描述 ---
    infos_path = "/home/ubuntu/ken/ReKV/infos.json"
    try:
        with open(infos_path, "r", encoding="utf-8") as f:
            info_map = json.load(f)
        print(f"[INFO] 成功加载 infos.json，共 {len(info_map)} 条描述。", flush=True)
    except Exception as e:
        print(f"[WARN] 读取 {infos_path} 失败：{e}", flush=True)
        info_map = {}

    # 注入到 personalize_set
    for name in personalize_set:
        if name in info_map and isinstance(info_map[name], str):
            personalize_set[name]['info'] = info_map[name]
        else:
            # 若无对应描述，则保持 None 或使用一个简易默认
            personalize_set[name]['info'] = f"Examples of {name}."

    print(">>> personalize_set 完成注入 info。", flush=True)
    for k, v in personalize_set.items():
        print(f"  {k}: info={v['info']}", flush=True)

    # 2. 正/负样本列表（带路径）
    qa_pos = []  # [(concept, PIL.Image, path)]
    qa_neg = []  # [(concept, PIL.Image, path)]
    test_root = './yollava-data/test'
    concepts = [c for c in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, c))]
    print(f">>> Building positive set from test ({len(concepts)} concepts)...", flush=True)
    for ci, concept in enumerate(concepts, 1):
        cdir = os.path.join(test_root, concept)
        for fn in os.listdir(cdir):
            p = os.path.join(cdir, fn)
            try:
                qa_pos.append((concept, Image.open(p).convert("RGB"), p))
            except Exception:
                pass
        if ci % 5 == 0 or ci == len(concepts):
            print(f"    Positive progress: {ci}/{len(concepts)} concepts", flush=True)

    print(">>> Building negative set (this may take a while)...", flush=True)
    for ti, target in enumerate(concepts, 1):
        count_before = len(qa_neg)
        for other in concepts:
            if other == target:
                continue
            odir = os.path.join(test_root, other)
            for fn in os.listdir(odir):
                p = os.path.join(odir, fn)
                try:
                    qa_neg.append((target, Image.open(p).convert("RGB"), p))
                except Exception:
                    pass
        added = len(qa_neg) - count_before
        print(f"    Negative progress: target={target} done, added {added} samples ({ti}/{len(concepts)})", flush=True)

    print(">>> personalize_set ready.")
    for k, v in personalize_set.items():
        print(f"  {k}: {len(v['images'])} train images", flush=True)
    print(f"正样本数量: {len(qa_pos)}, 负样本数量: {len(qa_neg)}", flush=True)

    # 3. 加载模型并注入
    model_path = "model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf"
    print(">>> Loading model...", flush=True)
    model, processor = llava_onevision_rekv.load_model(
        model_path=model_path,
        n_local=7500,
        topk=24,      # 调大一点召回，对正样本更友好；如负样本下滑可降回 16~20
        chunk_size=1
    )
    print(">>> Model loaded.", flush=True)

    model.clear_cache()
    model.encode_init_prompt()
    setup_retrieval_capture(model)

    print(">>> Injecting concepts into KV-Cache...", flush=True)
    for i, (name_, content) in enumerate(personalize_set.items(), 1):
        pair = {
            "id": name_,
            "category": content['category'],
            "images": content["images"],
            "text": content['info']
        }
        model.encode_personalized_pair(pair)
        if i % 5 == 0 or i == len(personalize_set):
            print(f"    Injected {i}/{len(personalize_set)}", flush=True)
    print(">>> Injection finished.\n", flush=True)

    # 4. 正样本推理并打印
    pos_correct = 0
    print("=== 正样本预测（期望 Yes）=== ", flush=True)
    for idx, (concept, img, path) in enumerate(qa_pos, 1):
        # 将 info 融入问句，并强约束只输出 A/B
        desc = personalize_set.get(concept, {}).get("info", "")
        if isinstance(desc, str) and len(desc) > 0:
            q = (
                f"Answer ONLY with 'A' or 'B'. A=Yes, B=No.\n"
                f"Say 'A' only if the image clearly contains <{concept}> "
                f"with the following characteristics: {desc}\n"
                f"Is <{concept}> in this photo?"
            )
        else:
            q = (
                f"Answer ONLY with 'A' or 'B'. A=Yes, B=No.\n"
                f"Say 'A' only if the image clearly contains <{concept}>.\n"
                f"Is <{concept}> in this photo?"
            )

        choice = run_inference(model, img, q)
        pred = 'A' if choice.startswith('A') else 'B'
        human = "Yes" if pred == 'A' else "No"
        if pred == 'A':
            pos_correct += 1
        print(f"[POS][{concept}][{os.path.basename(path)}] -> choice={choice} pred={human}", flush=True)
        if idx % 50 == 0:
            print(f"    Positive inference progress: {idx}/{len(qa_pos)}", flush=True)

    positive_acc = pos_correct / len(qa_pos) if qa_pos else 0.0
    print(f"正样本准确率: {positive_acc:.3f} ({pos_correct}/{len(qa_pos)})\n", flush=True)

    # 5. 负样本推理并打印
    neg_correct = 0
    print("=== 负样本预测（期望 No）=== ", flush=True)
    for idx, (concept, img, path) in enumerate(qa_neg, 1):
        desc = personalize_set.get(concept, {}).get("info", "")
        if isinstance(desc, str) and len(desc) > 0:
            q = (
                f"Answer ONLY with 'A' or 'B'. A=Yes, B=No.\n"
                f"Say 'A' only if the image clearly contains <{concept}> "
                f"with the following characteristics: {desc}\n"
                f"Is <{concept}> in this photo?"
            )
        else:
            q = (
                f"Answer ONLY with 'A' or 'B'. A=Yes, B=No.\n"
                f"Say 'A' only if the image clearly contains <{concept}>.\n"
                f"Is <{concept}> in this photo?"
            )

        choice = run_inference(model, img, q)
        pred = 'A' if choice.startswith('A') else 'B'
        human = "Yes" if pred == 'A' else "No"
        if pred == 'B':
            neg_correct += 1
        print(f"[NEG][ask:{concept}][file:{os.path.basename(path)}] -> choice={choice} pred={human}", flush=True)
        if idx % 200 == 0:
            print(f"    Negative inference progress: {idx}/{len(qa_neg)}", flush=True)

    negative_acc = neg_correct / len(qa_neg) if qa_neg else 0.0
    weighted_acc = 0.5 * (positive_acc + negative_acc)

    print("\n=== Final Recognition Accuracy ===", flush=True)
    print(f"Positive: {positive_acc:.3f} ({pos_correct}/{len(qa_pos)})", flush=True)
    print(f"Negative: {negative_acc:.3f} ({neg_correct}/{len(qa_neg)})", flush=True)
    print(f"Weighted: {weighted_acc:.3f}", flush=True)
    print(">>> Done.", flush=True)
