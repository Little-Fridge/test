#!/usr/bin/env python3
"""
用于测试yollava-text-qa数据集的个性化测试
"""

import torch
import argparse
import json
import sys
import os
from pathlib import Path
import types

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import llava_onevision_rekv
from decord import VideoReader, cpu
import numpy as np
from PIL import Image

'''
personalize_set = {"sks": 
    {
    "images": [PIL.Image.Image],
    "info": "A close-up of a vintage car's front bumper.",
    "category": "car"
    }
}
'''

# 用于保存检索信息的全局变量
last_retrieved_indices = None


def setup_retrieval_capture(model):
    """设置检索信息捕获"""
    global last_retrieved_indices
    last_retrieved_indices = None
    
    # 检查kv_cache是否已创建
    if not hasattr(model, 'kv_cache') or model.kv_cache is None:
        print("警告: kv_cache未创建，无法设置检索信息捕获")
        return
    
    # 对每个层的KV cache进行monkey patching
    for i, layer_kv in enumerate(model.kv_cache):
        if hasattr(layer_kv, 'reset_retrieval'):
            # 保存原始方法
            original_reset = layer_kv.reset_retrieval
            
            # 创建包装函数
            def make_wrapped_reset(original_func, kv_layer):
                def wrapped_reset():
                    global last_retrieved_indices
                    # 在重置之前保存检索信息
                    if hasattr(kv_layer, 'retrieved_block_indices') and kv_layer.retrieved_block_indices is not None:
                        if isinstance(kv_layer.retrieved_block_indices, list):
                            last_retrieved_indices = [idx.copy() if isinstance(idx, list) else idx for idx in kv_layer.retrieved_block_indices]
                        else:
                            last_retrieved_indices = kv_layer.retrieved_block_indices
                    else:
                        last_retrieved_indices = None
                    # 调用原始的reset_retrieval方法
                    return original_func()
                return wrapped_reset
            
            # 替换方法
            layer_kv.reset_retrieval = make_wrapped_reset(original_reset, layer_kv)


def get_last_retrieved_indices():
    """获取最后一次检索的索引"""
    return last_retrieved_indices

def test_personalization(personalize_set, qa, model_name="llava_ov_7b"):
    model_path = f'model_zoo/{model_name}'
    
    model, processor = llava_onevision_rekv.load_model(
        model_path=model_path,
        n_local=4000,    # 本地窗口大小
        topk=8,          # 检索块数
        chunk_size=1     # 块大小
    )

    model.clear_cache()
    model.encode_init_prompt()
    # 设置检索信息捕获（在kv_cache创建后）
    setup_retrieval_capture(model)

    # 编码个性化数据集
    print(f"===============编码个性化数据集===============")
    print(f"Personalization pairs num: {len(personalize_set)}")
    for name_, content in personalize_set.items():
        print(f"encoding {name_}...")
        pair = {
            "id": name_,
            "category": content['category'],
            "images": content["images"],
            "text": content['info']
        }
        model.encode_personalized_pair(pair)

        memory_usage = model.calc_memory_usage() / (1024**3)
        print(f"KV-Cache memory usage: {memory_usage:.2f} GB")
        
    # 测试text QA
    correct_num = 0
    corrent_num = 0
    for qa_ in qa:
        input_text = {
            "question": qa_['question'].replace('<sks>',qa_['name']),
            "prompt": model.get_choosing_prompt(qa_['question'].replace('<sks>',qa_['name']),qa_['options'],mc=True)
        }
        
        print(f"===============VQA测试===============")
        print(f"question: {input_text['question']}")
        print(f"prompt: {input_text['prompt']}")
        
        # answer = model.visual_question_answering(qa_['image'], input_text, max_new_tokens=128)
        answer = model.question_answering(input_text, max_new_tokens=128)

        print(f"回答: {answer}")
        print(f"答案: {qa_['correct_answer']}")
        if answer[0] == qa_['correct_answer']:
            correct_num += 1
        corrent_num += 1
        print(f"correct_num: {correct_num}, corrent_num: {corrent_num}, accuracy: {correct_num/corrent_num}")
        
        # 获取检索信息
        retrieved_indices = get_last_retrieved_indices()
        if retrieved_indices is not None:
            print(f"检索到的块索引: {retrieved_indices}")
        else:
            print("未检索到相关信息")
    
    return answer


if __name__ == "__main__":
    
    json_path = './yollava-data/text_qa.json'
    ds =  json.loads(open(json_path).read())

    
    personalize_set = {}
    qa =[]
    for name in ds:
        personalize_set[name] = {}
        personalize_set[name]['category'] = None
        personalize_set[name]['info'] = None
        personalize_set[name]['images'] = []
        for filename in os.listdir(f'./yollava-data/train/{name}'):
            image_path = f'./yollava-data/train/{name}/{filename}'
            personalize_set[name]['images'].append(Image.open(image_path))

        for j in ds[name]:
            qa.append({
                "name": name,
                "question": ds[name][j]['question'],
                "number": j,
                "options": ds[name][j]['option'],
                "correct_answer": ds[name][j]['correct_answer']
            })

    print("个性化数据集已加载，包含以下项目:")
    for k, v in personalize_set.items():
        print(f"name: {k}, category: {v['category']}, info: {v['info']}, images num: {len(v['images'])}")
    print("图片已经加载，开始个性化测试...")
    
    
    

    test_personalization(
        personalize_set=personalize_set,
        qa=qa,
        model_name="LLaVA/llava-onevision-qwen2-7b-ov-hf"
    )