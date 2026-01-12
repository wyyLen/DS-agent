from typing import List, Optional

import numpy as np
import requests

from dsagent_core.logs import logger


class CustomEmbeddingComparisonEngine:
    def __init__(self):
        self.embedding_cache = {}

    def get_embedding(self, text: str, embedding_config: dict) -> List[float]:
        # desc: 通过自定义模型服务获取文本的embedding向量,返回文本的embedding向量

        # 生成缓存键，包含文本及模型相关配置
        config_key = (
            embedding_config.get("model_name", ""),
            embedding_config.get("base_url", ""),
        )
        cache_key = (text, config_key)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            response = requests.post(
                embedding_config["base_url"],
                json={
                    "model": embedding_config["model_name"],
                    "input": [text],
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": embedding_config["api_key"] if "api_key" in embedding_config else "",
                },
                timeout=10
            )
            response.raise_for_status()
            # openai的embedding和ollama的格式不一样
            if "api_key" in embedding_config:
                embeddings_result = response.json()["data"]["embeddings"]
            else:
                embeddings_result = response.json()["embeddings"]

            # 处理二维数组响应
            embedding = embeddings_result[0] if isinstance(embeddings_result[0], list) else embeddings_result
            embedding = [float(x) for x in embedding]
            self.embedding_cache[cache_key] = embedding
            return embedding

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding request failed: {str(e)}")
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"Invalid response format: {str(e)}")

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        # 返回：余弦相似度得分，范围[-1, 1]
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)
        norm_product = np.linalg.norm(np_vec1) * np.linalg.norm(np_vec2)
        return np.dot(np_vec1, np_vec2) / norm_product if norm_product != 0 else 0.0

    def run(self, text1: str, text2: str, embedding_config: dict) -> Optional[float]:
        try:
            emb1 = self.get_embedding(text1, embedding_config)
            emb2 = self.get_embedding(text2, embedding_config)
            if len(emb1) != len(emb2):
                raise RuntimeError("Embedding dimension mismatch")
            return self.cosine_similarity(emb1, emb2)
        except RuntimeError as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return None