import random
from collections import Counter
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from dsagent_core.rag.engines.GraphMatching.graph import GraphSet
from dsagent_core.rag.engines.customEmbeddingComparisonEngine import CustomEmbeddingComparisonEngine
from metagpt.schema import Task, Plan


class NodeSimilarityCalculator:
    def __init__(self, walk_length: int = 3, num_walks: int = 3, alpha: float = 0.8, beta: float = 0.5,
                 embedding_dim: int = 768):
        # 参数： walk_length: 随机游走的长度, num_walks: 每个节点的随机游走次数
        #       alpha: 转移概率系数, beta: 距离衰减系数, embedding_dim: 嵌入向量的维度
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.alpha = alpha
        self.beta = beta
        self.embedding_dim = embedding_dim
        self.node_embeddings = {}  # 存储节点嵌入向量的缓存

    def calculate_similarity(self, graph1: GraphSet, node1: str, graph2: GraphSet, node2: str) -> float:
        """
        计算两个工作流图中两个节点的相似度
        参数:
            graph1: 第一个工作流图, node1: 第一个图中的节点编号
            graph2: 第二个工作流图, node2: 第二个图中的节点编号
        返回:
            两个节点的相似度分数(0-1之间)
        """
        embedding1 = self._get_node_embedding(graph1, node1)
        embedding2 = self._get_node_embedding(graph2, node2)
        similarity = self._cosine_similarity(embedding1, embedding2)
        return max(0.0, min(1.0, similarity))  # 确保结果在[0,1]范围内

    def _get_node_embedding(self, graph: GraphSet, node: str) -> np.ndarray:
        # 获取节点的嵌入向量，如果未缓存则生成
        # graph: 工作流图, node: 节点编号
        cache_key = f"{id(graph)}_{node}"
        if cache_key in self.node_embeddings:
            return self.node_embeddings[cache_key]
        walks = self._random_walks(graph, node)
        embedding = self._generate_embedding(walks)
        self.node_embeddings[cache_key] = embedding
        return embedding

    def _random_walks(self, graph: GraphSet, start_node: str) -> List[List[str]]:
        # 执行基于动态转移概率的随机游走
        # 参数: graph: 工作流图, start_node: 起始节点编号
        walks = []
        offset = 0
        for _ in range(self.num_walks):
            walk = [start_node]
            current_node = start_node
            current_distance = 0  # 与起始节点的拓扑距离
            while len(walk) < self.walk_length:
                neighbor_edges = graph.neighbor(offset, int(current_node) - 1)
                neighbors = []
                for edge in neighbor_edges:
                    v1, v2 = edge.split(":")
                    neighbor = v1 if v2 == current_node else v2
                    neighbors.append(neighbor)
                if not neighbors:
                    break
                # 计算转移概率，考虑距离衰减
                probs = []
                for neighbor in neighbors:
                    prob = self.alpha / (1 + self.beta * current_distance)
                    probs.append(prob)
                # 归一化概率
                sum_probs = sum(probs)
                if sum_probs == 0:
                    norm_probs = [1.0 / len(probs)] * len(probs)  # 均匀分布
                else:
                    norm_probs = [p / sum_probs for p in probs]
                # 根据概率选择下一个节点
                next_node = random.choices(neighbors, weights=norm_probs, k=1)[0]
                walk.append(next_node)
                current_node = next_node
                current_distance += 1
            walks.append(walk)
        return walks

    def _generate_embedding(self, walks: List[List[str]]) -> np.ndarray:
        # 从随机游走路径生成节点嵌入向量
        embedding = np.zeros(self.embedding_dim)
        count = 0
        for walk in walks:
            for node in walk:
                embedding_engine = CustomEmbeddingComparisonEngine()
                local_embedding_config = dict(
                    base_url="http://192.168.3.88:11434/api/embed",
                    model_name="nomic-embed-text:latest"
                )
                openai_embedding_config = dict(
                    base_url="https://us.ifopen.ai/v1/embeddings",
                    model_name="text-embedding-3-small",
                    api_key="xxx"
                )
                server_embedding_config = dict(
                    base_url="http://60.245.208.139:11434//api/embed",
                    model_name="nomic-embed-text:latest"
                )
                walk_embedding = embedding_engine.get_embedding(node, local_embedding_config)
                embedding += walk_embedding
                count += 1
        if count > 0:
            embedding /= count  # 均值聚合多路径语义信息
        return embedding

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # 返回：余弦相似度得分，范围[-1, 1]
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)
        norm_product = np.linalg.norm(np_vec1) * np.linalg.norm(np_vec2)
        return np.dot(np_vec1, np_vec2) / norm_product if norm_product != 0 else 0.0


def _json2plan(tasks: list):
    task_list: list[Task] = []
    for task in tasks:
        task_list.append(Task(task_id=task["task_id"], instruction=task["instruction"],
                              dependent_task_ids=task["dependent_task_ids"], task_type=task["task_type"]))
    plan = Plan(goal="", tasks=task_list)
    return plan


def _getGraphEditDistance(graph1: GraphSet, graph2: GraphSet):
    n, m = len(graph1.curVSet(0)), len(graph2.curVSet(0))
    matrix = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            dis = 0
            if graph1.curVSet(0).get(str(i + 1))["task_type"] != graph2.curVSet(0).get(str(j + 1))["task_type"]:
                dis += 1
            # node_sim_engine = NodeSimilarityCalculator(walk_length=3, num_walks=3, alpha=0.8, beta=0.5, embedding_dim=768)
            # dis -= node_sim_engine.calculate_similarity(graph1, str(i + 1), graph2, str(j + 1))
            g1_neighbors = graph1.neighbor_with_type(0, i)
            g2_neighbors = graph2.neighbor_with_type(0, j)
            g1_before, g1_after = g1_neighbors.get('before'), g1_neighbors.get('after')
            g2_before, g2_after = g2_neighbors.get('before'), g2_neighbors.get('after')

            def list_difference(list1: list[dict], list2: list[dict]):
                types1 = [item["task_type"] for item in list1]
                types2 = [item["task_type"] for item in list2]
                count1 = Counter(types1)
                count2 = Counter(types2)
                difference = 0
                for element in count1:
                    diff = abs(count1[element] - count2.get(element, 0))
                    difference += diff
                for element in count2:
                    if element not in count1:
                        difference += count2[element]
                return difference

            matrix[i][j] = dis + list_difference(g1_before, g2_before) + list_difference(g1_after, g2_after)
    row_ind, col_ind = linear_sum_assignment(matrix)
    edit_distance = sum(matrix[i][j] for i, j in zip(row_ind, col_ind))
    return edit_distance