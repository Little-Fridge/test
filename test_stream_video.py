#!/usr/bin/env python3
"""
简单的流式视频测试脚本
用于测试一条流视频和一条prompt的ReKV功能
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


def load_video(video_path, sample_fps=1.0):
    """加载视频并采样帧"""
    print(f"正在加载视频: {video_path}")
    
    if video_path.endswith('.npy'):
        video = np.load(video_path)
        assert sample_fps <= 1
        num_frames = len(video)
        frame_idx = np.linspace(0, num_frames-1, int(num_frames*sample_fps), dtype=int).tolist()
        video = video[frame_idx]
    else:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), int(fps / sample_fps))]
        video = vr.get_batch(frame_idx).asnumpy()
    
    print(f"视频形状: {video.shape}")
    return video


def test_stream_video(video_path, prompt, model_name="llava_ov_7b", sample_fps=1.0, stream_chunk_size=16):
    """测试流式视频处理"""
    
    print("=" * 60)
    print("ReKV 流式视频测试")
    print("=" * 60)
    print(f"视频路径: {video_path}")
    print(f"提示词: {prompt}")
    print(f"模型: {model_name}")
    print(f"采样帧率: {sample_fps}")
    print(f"流式块大小: {stream_chunk_size}")
    print("=" * 60)
    
    # 1. 加载ReKV模型
    print("\n1. 正在加载ReKV模型...")
    model_path = f'model_zoo/{model_name}'
    
    model, processor = llava_onevision_rekv.load_model(
        model_path=model_path,
        n_local=4000,    # 本地窗口大小
        topk=8,          # 检索块数 (减少到8个，使检索效果更明显)
        chunk_size=1     # 块大小
    )
    
    print("ReKV模型加载完成!")
    print(f"本地窗口大小: {model.n_local}")
    print(f"检索块数: {model.topk}")
    
    # 2. 加载视频
    print("\n2. 正在加载视频...")
    video = load_video(video_path, sample_fps)
    video_tensor = torch.from_numpy(video)
    print("视频加载完成!")
    
    # 3. 初始化ReKV
    print("\n3. 正在初始化ReKV...")
    model.clear_cache()
    model.encode_init_prompt()
    print("ReKV初始化完成!")
    
    # 设置检索信息捕获（在kv_cache创建后）
    setup_retrieval_capture(model)
    
    # 4. 流式处理视频
    print("\n4. 开始流式处理视频...")
    print("这个过程会模拟流式视频输入，逐块处理")
    
    num_frames = len(video)
    num_chunks = (num_frames + stream_chunk_size - 1) // stream_chunk_size
    
    print(f"总帧数: {num_frames}")
    print(f"块数: {num_chunks}")
    print(f"每块帧数: {stream_chunk_size}")
    
    # 模拟流式输入
    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * stream_chunk_size
        end_frame = min(start_frame + stream_chunk_size, num_frames)
        
        print(f"\n--- 处理块 {chunk_idx + 1}/{num_chunks} (帧 {start_frame}-{end_frame-1}) ---")
        
        # 获取当前块的视频
        chunk_video = video_tensor[start_frame:end_frame]
        print(f"当前块形状: {chunk_video.shape}")
        
        # 使用ReKV编码当前块
        model.encode_video(chunk_video, encode_chunk_size=8)
        
        # 显示当前内存使用
        memory_usage = model.calc_memory_usage() / (1024**3)
        print(f"当前KV-Cache内存使用: {memory_usage:.2f} GB")
        
        # 添加更详细的内存信息
        if hasattr(model, 'kv_cache') and model.kv_cache is not None:
            if hasattr(model.kv_cache[0], 'length'):
                total_tokens = model.kv_cache[0].length
                print(f"总token数: {total_tokens}")
                print(f"本地窗口大小: {model.n_local}")
                print(f"是否触发offload: {total_tokens > model.n_local + 13}")
                if hasattr(model.kv_cache[0], 'num_global_block'):
                    print(f"CPU中的全局块数: {model.kv_cache[0].num_global_block}")
        
        # 如果处理了足够的帧，可以进行问答测试
        if chunk_idx >= 0:  # 可以从第一个块开始测试
            print(f"测试问答 (基于前 {end_frame} 帧)...")
            
            # 显示检索状态
            total_blocks = model.kv_cache[0].num_global_block if hasattr(model.kv_cache[0], 'num_global_block') else 0
            retrieval_enabled = total_blocks > model.topk
            print(f"检索模式: {'启用' if retrieval_enabled else '禁用'} (总块数: {total_blocks}, topk: {model.topk})")
            
            input_text = {
                "question": prompt,
                "prompt": model.get_prompt(prompt)
            }
            
            answer = model.question_answering(input_text, max_new_tokens=128)
            print(f"基于前 {end_frame} 帧的回答: {answer}")
            
            # 获取检索到的帧序号
            retrieved_indices = get_last_retrieved_indices()
            if retrieved_indices is not None and len(retrieved_indices) > 0:
                # 转换块索引为帧序号 (每个块对应一帧)
                retrieved_frame_indices = retrieved_indices[0] if isinstance(retrieved_indices, list) and len(retrieved_indices) > 0 else retrieved_indices
                print(f"检索到的帧序号: {retrieved_frame_indices}")
            else:
                print("检索到的帧序号: 无 (使用所有帧或检索未启用)")
    
    # 5. 最终问答（基于完整视频）
    print("\n5. 最终问答 (基于完整视频)...")
    
    # 显示最终检索状态
    total_blocks = model.kv_cache[0].num_global_block if hasattr(model.kv_cache[0], 'num_global_block') else 0
    retrieval_enabled = total_blocks > model.topk
    print(f"最终检索模式: {'启用' if retrieval_enabled else '禁用'} (总块数: {total_blocks}, topk: {model.topk})")
    
    input_text = {
        "question": prompt,
        "prompt": model.get_prompt(prompt)
    }
    
    final_answer = model.question_answering(input_text, max_new_tokens=256)
    
    # 显示最终问答的检索信息
    print(f"最终回答: {final_answer}")
    retrieved_indices = get_last_retrieved_indices()
    if retrieved_indices is not None and len(retrieved_indices) > 0:
        # 转换块索引为帧序号 (每个块对应一帧)
        retrieved_frame_indices = retrieved_indices[0] if isinstance(retrieved_indices, list) and len(retrieved_indices) > 0 else retrieved_indices
        print(f"最终问答检索到的帧序号: {retrieved_frame_indices}")
    else:
        print("最终问答检索到的帧序号: 无 (使用所有帧或检索未启用)")
    
    # 6. 输出结果
    print("\n" + "=" * 60)
    print("流式视频测试结果")
    print("=" * 60)
    print(f"视频总帧数: {num_frames}")
    print(f"处理块数: {num_chunks}")
    print(f"提示词: {prompt}")
    print(f"最终KV-Cache内存使用: {model.calc_memory_usage() / (1024**3):.2f} GB")
    print("=" * 60)
    
    return {
        "video_path": video_path,
        "prompt": prompt,
        "model": model_name,
        "sample_fps": sample_fps,
        "stream_chunk_size": stream_chunk_size,
        "total_frames": num_frames,
        "num_chunks": num_chunks,
        "final_answer": final_answer,
        "final_memory_usage_gb": model.calc_memory_usage() / (1024**3)
    }


def main():
    parser = argparse.ArgumentParser(description="ReKV流式视频测试")
    parser.add_argument("--video_path", type=str, required=True, 
                       help="视频文件路径")
    parser.add_argument("--prompt", type=str, required=True,
                       help="要问的问题或提示词")
    parser.add_argument("--model", type=str, default="llava-onevision-qwen2-0.5b-ov-hf",
                       help="要使用的模型")
    parser.add_argument("--sample_fps", type=float, default=1.0,
                       help="视频采样帧率")
    parser.add_argument("--stream_chunk_size", type=int, default=16,
                       help="流式处理的块大小（帧数）")
    
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件不存在: {args.video_path}")
        return
    
    # 运行流式视频测试
    try:
        result = test_stream_video(
            video_path=args.video_path,
            prompt=args.prompt,
            model_name=args.model,
            sample_fps=args.sample_fps,
            stream_chunk_size=args.stream_chunk_size
        )
        
        # 保存结果
        output_file = "stream_video_test_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n流式视频测试结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"流式视频测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()