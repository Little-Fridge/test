import os
import subprocess
import shutil
import json
from pathlib import Path


def clone_or_pull(repo_url, target_dir):
    if os.path.exists(target_dir):
        print(f"{target_dir} exists, pulling latest...")
        subprocess.run(['git', '-C', target_dir, 'pull'])
    else:
        print(f"Cloning {repo_url} into {target_dir} ...")
        subprocess.run(['git', 'clone', repo_url, target_dir])


def move_file(src_pattern, dst_file):
    import glob
    files = glob.glob(src_pattern)
    if files:
        shutil.move(files[0], dst_file)
        print(f"Moved {files[0]} -> {dst_file}")
    else:
        print(f"File not found for pattern: {src_pattern}")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ----------- 下载模型 -----------
model_repos = {
    "llava-onevision-qwen2-0.5b-ov-hf": "https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    # "llava-onevision-qwen2-7b-ov-hf": "https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf",
    # "llava-onevision-qwen2-72b-ov-hf": "https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf",
    # "LongVA-7B": "https://huggingface.co/lmms-lab/LongVA-7B",
    # "Video-LLaVA-7B-hf": "https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf",
}

ensure_dir("model_zoo")
for name, url in model_repos.items():
    clone_or_pull(url, os.path.join("model_zoo", name))


print("\n下载完成！")
