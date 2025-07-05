import os
import requests
from tqdm import tqdm
import zipfile

# 下载文件函数


def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"{save_path} 已存在，跳过下载。")
        return
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as file, tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    print(f"已下载: {save_path}")


# 创建保存目录
save_dir = 'data/qaego4d'
os.makedirs(save_dir, exist_ok=True)

# 文件链接和保存路径
files = [
    {
        "url": "https://huggingface.co/datasets/Becomebright/QAEgo4D-MC-test/resolve/main/test_mc.json",
        "save_path": os.path.join(save_dir, "test_mc.json")
    },
    {
        "url": "https://huggingface.co/datasets/Becomebright/QAEgo4D-MC-test/resolve/main/videos.zip",
        "save_path": os.path.join(save_dir, "videos.zip")
    }
]

# 下载文件
for file in files:
    download_file(file["url"], file["save_path"])

# 解压视频
zip_path = os.path.join(save_dir, "videos.zip")
extract_dir = os.path.join(save_dir, "videos")
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")
else:
    print(f"{zip_path} 不存在，无法解压。")
