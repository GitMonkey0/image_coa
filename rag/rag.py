import torch
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.spatial.distance import cosine
from transformers import Blip2Processor, Blip2VisionModelWithProjection
from tqdm import tqdm
import os
import json

def get_file_paths(directory):
    """获取目录下所有文件的绝对路径"""
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.abspath(os.path.join(root, filename))
            file_paths.append(filepath)
    return file_paths

def compute_image_similarity(image_paths, model_name="../../ckpt/blip2-opt-2.7b", k=5):
    """计算图像列表中各图像间的相似度"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2VisionModelWithProjection.from_pretrained(model_name).to(device)
    model.eval()  # 设置为评估模式

    # 生成图像embedding
    image_embeds = []
    for path in tqdm(image_paths, desc="生成图像特征"):
        try:
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embed = outputs.image_embeds.cpu().numpy()[0]
            image_embeds.append(embed)
        except Exception as e:
            print(f"处理图像 {path} 时出错: {e}")
            image_embeds.append(None)  # 出错时添加None占位

    # 过滤掉处理失败的图像
    valid_indices = [i for i, embed in enumerate(image_embeds) if embed is not None]
    valid_paths = [image_paths[i] for i in valid_indices]
    valid_embeds = np.array([image_embeds[i] for i in valid_indices])

    # 计算相似度矩阵
    n = len(valid_paths)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = -1  # 排除自身对比
            else:
                similarity_matrix[i][j] = 1 - cosine(valid_embeds[i], valid_embeds[j])

    # 获取每个图像的前k相似图像
    result = {}
    for i, path in enumerate(valid_paths):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        top_k_indices = sorted_indices[:k]
        top_k_pairs = [
            (valid_paths[j], float(similarity_matrix[i][j]))
            for j in top_k_indices
        ]
        result[path] = top_k_pairs

    return result

def save_results(result_dict, output_path):
    """保存相似度结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    # 设置参数
    IMAGE_DIR = "../../data/UnsafeBench/data/train_images"  # 图像目录
    OUTPUT_FILE = "../../data/UnsafeBench/exp/image_similarity_results.json"  # 输出文件
    TOP_K = 5  # 取前5相似的图像

    # 执行流程
    print(f"正在扫描目录: {IMAGE_DIR}")
    image_paths = get_file_paths(IMAGE_DIR)
    print(f"找到 {len(image_paths)} 张图像")

    if image_paths:
        similarity_dict = compute_image_similarity(image_paths, k=TOP_K)
        save_results(similarity_dict, OUTPUT_FILE)
    else:
        print("未找到图像文件，请检查目录路径")