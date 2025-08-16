#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rekv_personalization_text.py

用于测试 yollava-text-qa 数据集的个性化测试（文本多选），
从已有的 infos.json 直接读取 info 文本进行注入。
对齐你原项目的“分块注入 + 检索”，并只打印整数块索引。
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image

# 把项目根目录加入 Python 路径
sys.path.append(str(Path(__file__).resolve().parent))

from model import llava_onevision_rekv  # noqa: E402


# ============================== KV 占用：安全统计 ==============================

def _tensor_bytes(x: Any) -> int:
    if torch.is_tensor(x):
        try:
            return x.numel() * x.element_size()
        except Exception:
            return 0
    return 0


def _iter_member_objects(obj: Any) -> Iterable[Any]:
    if obj is None:
        return
    if isinstance(obj, (list, tuple, set)):
        for it in obj:
            yield it
        return
    if isinstance(obj, dict):
        for it in obj.values():
            yield it
        return
    try:
        names = dir(obj)
    except Exception:
        names = []
    for name in names:
        if name.startswith("_"):
            continue
        low = name.lower()
        if not any(k in low for k in ("cache", "k", "v", "kv", "key", "value", "index", "indices", "block")):
            continue
        try:
            yield getattr(obj, name)
        except Exception:
            continue


def _walk_bytes(obj: Any, seen: set) -> int:
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if torch.is_tensor(obj):
        return _tensor_bytes(obj)
    total = 0
    for child in _iter_member_objects(obj):
        total += _walk_bytes(child, seen)
    return total


def get_kv_cache_gb(model) -> float:
    total = 0
    seen: set = set()
    if hasattr(model, "kv_cache") and model.kv_cache is not None:
        for layer in model.kv_cache:
            total += _walk_bytes(layer, seen)
    return total / (1024 ** 3)


# ============================== 检索捕获：只打印块索引 ==============================

_retrieval_hits: List[Dict[str, Any]] = []  # [{'layer': i, 'indices': [..]}, ...]


def _is_int_like_list(x) -> bool:
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(i, int) for i in x)


def _pick_indices_from_layer(layer) -> List[List[int]]:
    hits: List[List[int]] = []
    keys = ("retriev", "indices", "index", "blocks", "block_indices")
    for name in dir(layer):
        if name.startswith("_"):
            continue
        low = name.lower()
        if not any(k in low for k in keys):
            continue
        try:
            val = getattr(layer, name)
        except Exception:
            continue
        if torch.is_tensor(val) and val.dtype in (torch.int64, torch.int32):
            try:
                hits.append(val.detach().cpu().reshape(-1).tolist())
            except Exception:
                pass
        elif _is_int_like_list(val):
            hits.append(list(val))
    return hits


def setup_retrieval_capture(model):
    """在 kv 层 reset_retrieval 上打钩，捕获“最后一次检索到的块索引”（仅整数）"""
    global _retrieval_hits
    _retrieval_hits = []

    if not hasattr(model, "kv_cache") or model.kv_cache is None:
        print("警告: kv_cache未创建，无法设置检索信息捕获")
        return

    for li, layer_kv in enumerate(model.kv_cache):
        if not hasattr(layer_kv, "reset_retrieval"):
            continue
        orig = layer_kv.reset_retrieval

        def make_wrapped(_orig, _layer, _li):
            def _wrapped():
                global _retrieval_hits
                layer_hits = _pick_indices_from_layer(_layer)
                if layer_hits:
                    first = layer_hits[0]
                    if isinstance(first, list) and len(first) > 16:
                        first = first[:16]
                    _retrieval_hits.append({"layer": _li, "indices": first})
                return _orig()
            return _wrapped

        layer_kv.reset_retrieval = make_wrapped(orig, layer_kv, li)


def get_last_retrieved_indices(model=None):
    if _retrieval_hits:
        return _retrieval_hits
    # 临时扫一遍（如果 hook 还没触发过）
    if model is None or not hasattr(model, "kv_cache") or model.kv_cache is None:
        return None
    tmp = []
    for li, layer in enumerate(model.kv_cache):
        layer_hits = _pick_indices_from_layer(layer)
        if layer_hits:
            first = layer_hits[0]
            if isinstance(first, list) and len(first) > 16:
                first = first[:16]
            tmp.append({"layer": li, "indices": first})
    return tmp or None


# ============================== 选项/答案处理 ==============================

def normalize_options(opt) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    支持 list[str] 或 dict{'A': 'xxx', 'B': 'yyy'}
    返回 (labels, contents, mapping_dict)
    """
    if isinstance(opt, dict):
        labels = sorted([k for k in opt.keys() if isinstance(k, str)], key=lambda s: s)
        contents = [str(opt[k]) for k in labels]
        mapping = {k: str(opt[k]) for k in labels}
        return labels, contents, mapping
    elif isinstance(opt, (list, tuple)):
        labels = [chr(ord('A') + i) for i in range(len(opt))]
        contents = [str(x) for x in opt]
        mapping = {labels[i]: contents[i] for i in range(len(labels))}
        return labels, contents, mapping
    else:
        raise TypeError(f"Unsupported options type: {type(opt)}")


_choice_pat = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def parse_choice(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = _choice_pat.search(text.strip())
    return m.group(1).upper() if m else ""


# ============================== 文本生成（不走视觉塔） ==============================

def text_generate(model, processor, prompt: str, max_new_tokens: int = 16) -> str:
    """
    纯文本问答：构造 chat 模板 → tokenizer → model.generate
    """
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor.tokenizer 不存在，无法进行文本生成。")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )

    device = next(model.parameters()).device
    chat_ids = chat_ids.to(device)

    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.eos_token_id,
    }

    out_ids = model.generate(input_ids=chat_ids, attention_mask=torch.ones_like(chat_ids), **gen_cfg)
    out_txt = tok.decode(out_ids[0][chat_ids.shape[-1]:], skip_special_tokens=True).strip()
    return out_txt


# ============================== 主流程 ==============================

@torch.inference_mode()
def test_personalization(personalize_set, qa, model_name):
    # 加载模型（使用你的本地路径）
    model, processor = llava_onevision_rekv.load_model(
        model_path=f"model_zoo/{model_name}",
        # 对齐你原项目：多块注入 + 检索
        n_local=7500,
        topk=8,
        chunk_size=49,      # 7x7 一个块
        block_size=49,      # 与 chunk_size 对齐
        n_frame_tokens=196, # 14x14 patch
    )

    # 初始化 ReKV：先建 cache 再挂 hook
    if hasattr(model, "clear_cache"):
        model.clear_cache()
    if hasattr(model, "_get_cache"):
        model._get_cache()
    if hasattr(model, "encode_init_prompt"):
        model.encode_init_prompt()
    setup_retrieval_capture(model)

    print("===============编码个性化数据集===============")
    for name_, content in personalize_set.items():
        print(f"encoding {name_} ...")
        pair = {
            "id": name_,
            "category": content['category'],
            "images": content["images"],  # List[PIL.Image.Image]
            "text": content['info'],
        }
        model.encode_personalized_pair(pair)
        print(f"  KV-Cache 使用内存: {get_kv_cache_gb(model):.2f} GB")
    torch.cuda.synchronize()

    # 如果有 pair 没写入块，兜底重编一次
    if getattr(model, "kv_cache", None):
        layer0 = model.kv_cache[0]
        if hasattr(layer0, "pairs"):
            redo = [pid for pid, info in layer0.pairs.items() if len(info.get("blocks", [])) == 0]
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

    print("\n=== VQA 多选测试（文本） ===")
    correct, total = 0, 0
    per_concept = defaultdict(lambda: {"correct": 0, "total": 0})

    for qa_ in qa:
        # 数据集中占位符可能是 "<sks>" 或 "sks"；两种都兼容
        q_text = qa_["question"].replace("<sks>", qa_["name"]).replace("sks", qa_["name"])
        labels, contents, mapping = normalize_options(qa_["options"])

        # 用模型内置的多选 Prompt 生成器（传入纯文本选项）
        prompt = model.get_choosing_prompt(q_text, contents, mc=True)

        print(f"\n[Q] {q_text}")
        print(f"prompt: {prompt}")

        # 打印上一次检索到的块索引（只整数，最多显示两层）
        hits = get_last_retrieved_indices(model)
        if hits:
            for h in hits[:2]:
                print(f"  检索到的块索引(层{h['layer']}): {h['indices']}")
        else:
            print("  检索到的块索引: None")

        # 纯文本推理
        answer_text = text_generate(model, processor, prompt, max_new_tokens=16)
        choice = parse_choice(answer_text)
        pred_text = mapping.get(choice, "")

        print(f"[Pred] {choice}) {pred_text}    [GT] {qa_['correct_answer']}")

        if choice.upper() == str(qa_["correct_answer"]).upper():
            correct += 1
            per_concept[qa_["name"]]["correct"] += 1
        per_concept[qa_["name"]]["total"] += 1
        total += 1

        print(f"   Acc: {correct}/{total} = {correct/total:.3f}")

    print(f"\n=== Final Accuracy: {correct}/{total} = {correct/total:.3f} ===")

    print("\n=== 每个概念的准确率 ===")
    for name in sorted(per_concept.keys()):
        c = per_concept[name]["correct"]
        t = per_concept[name]["total"]
        acc = (c / t) if t else 0.0
        print(f"{name}: {acc:.3f} ({c}/{t})")


def main():
    p = argparse.ArgumentParser(description="ReKV Text-QA 个性化测试（读取 infos.json 注入 info）")
    p.add_argument("--data_root", "-d", type=str, default="./yollava-data",
                   help="数据根目录，包含 train/ 和 text_qa.json")
    # 保持和你之前一致的模型名，最终拼接成 model_zoo/<model_name>
    p.add_argument("--model_name", "-m", type=str, default="LLaVA/llava-onevision-qwen2-7b-ov-hf")
    # infos.json 默认用你给的绝对路径；可改成相对路径：./infos.json
    p.add_argument("--infos", type=str, default="infos.json")
    args = p.parse_args()

    train_root = os.path.join(args.data_root, "train")
    test_json = os.path.join(args.data_root, "text_qa.json")
    ds = json.load(open(test_json, encoding="utf-8"))

    # 1) 构造 personalize_set（先不填 info）
    personalize_set: Dict[str, Dict[str, Any]] = {}
    for name, _items in ds.items():
        imgs = []
        tdir = os.path.join(train_root, name)
        for fn in sorted(os.listdir(tdir)):
            imgs.append(Image.open(os.path.join(tdir, fn)).convert("RGB"))
        personalize_set[name] = {
            "category": name,
            "images": imgs,
            "info": None,
        }

    # 2) 读取 infos.json，注入 info
    info_map = json.load(open(args.infos, encoding="utf-8"))
    print(f"[INFO] 从 {args.infos} 读取到 {len(info_map)} 条描述")
    for name in personalize_set:
        personalize_set[name]["info"] = info_map.get(name, "")

    # 3) 构造 QA 列表
    qa: List[Dict[str, Any]] = []
    for name, items in ds.items():
        for _, qa_item in items.items():
            qa.append({
                "name": name,
                "question": qa_item["question"],
                "options": qa_item["option"],   # 注意：数据里是 'option'（dict），不是 'options'
                "correct_answer": qa_item["correct_answer"],
            })

    # 4) 运行
    test_personalization(personalize_set, qa, args.model_name)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
