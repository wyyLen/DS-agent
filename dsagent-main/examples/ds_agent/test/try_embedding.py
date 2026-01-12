import numpy as np
import requests
from typing import List, Optional


def get_embedding(text: str, embedding_config: dict) -> List[float]:
    """
    通过自定义模型服务获取文本的embedding向量

    参数：
        text: 需要编码的文本
        model_url: 自定义embedding模型服务的URL

    返回：
        文本的embedding向量

    异常：
        RuntimeError: 当请求失败或响应格式异常时抛出
    """
    try:
        # 构造符合常见模型服务格式的请求（可根据需要修改请求体格式）
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(
            embedding_config["base_url"],
            json={
                "model": embedding_config["model_name"],
                "input": [text],
            },
            headers=headers,
            timeout=10
        )
        response.raise_for_status()  # 自动处理HTTP错误状态码

        # 解析响应并验证格式
        embeddings_result = response.json()["embeddings"]
        if not isinstance(embeddings_result, list) or not len(embeddings_result) > 0:
            raise ValueError("Unexpected response format")

        # 处理二维数组响应（常见于批量接口）
        embedding = embeddings_result[0] if isinstance(embeddings_result[0], list) else embeddings_result
        if not all(isinstance(x, (float, int)) for x in embedding):
            raise ValueError("Invalid embedding values")

        return [float(x) for x in embedding]

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Embedding request failed: {str(e)}")
    except (ValueError, KeyError) as e:
        raise RuntimeError(f"Invalid response format: {str(e)}")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度

    参数：
        vec1: 第一个向量
        vec2: 第二个向量

    返回：
        余弦相似度得分，范围[-1, 1]
    """
    np_vec1 = np.array(vec1)
    np_vec2 = np.array(vec2)

    dot_product = np.dot(np_vec1, np_vec2)
    norm_product = np.linalg.norm(np_vec1) * np.linalg.norm(np_vec2)

    # 处理零向量情况
    if norm_product == 0:
        return 0.0

    return dot_product / norm_product


def semantic_similarity(text1: str, text2: str, embedding_config: dict) -> Optional[float]:
    """
    计算两个文本的语义相似度

    参数：
        text1: 第一个文本
        text2: 第二个文本
        model_url: 自定义embedding模型服务URL

    返回：
        余弦相似度得分（None表示计算失败）
    """
    try:
        emb1 = get_embedding(text1, embedding_config)
        emb2 = get_embedding(text2, embedding_config)
        print(f"emb1: {emb1},\n emb2: {emb2}")

        if len(emb1) != len(emb2):
            raise RuntimeError("Embedding dimension mismatch")

        return cosine_similarity(emb1, emb2)

    except RuntimeError as e:
        print(f"Error calculating similarity: {str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    # 示例URL（替换为实际模型服务地址）
    embedding_config = dict(
        base_url="http://192.168.3.88:11434/api/embed",
        model_name="nomic-embed-text:latest"
    )

    text_a = "The quick brown fox jumps over the lazy dog"
    text_b = "A fast brown animal leaps over a sleeping canine"

    similarity = semantic_similarity(text_a, text_b, embedding_config)
    if similarity is not None:
        print(f"Semantic similarity: {similarity:.4f}")