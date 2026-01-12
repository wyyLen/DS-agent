import math
from typing import List
import numpy as np

from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    NodeWithScore,
    QueryBundle,
)
from llama_index.retrievers.bm25 import BM25Retriever

from metagpt.actions.ds_agent.query_utils import QueryUtils
from metagpt.const import EXAMPLE_DATA_PATH
from metagpt.logs import logger

EXP_PATH = EXAMPLE_DATA_PATH / "exp_bank/plan_exp.json"


class CustomMixtureRetriever(BaseRetriever):
    def __init__(
            self,
            vector_retriever: VectorIndexRetriever,
            bm25_retriever: BM25Retriever,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self.weight_faiss = 0.6
        self.weight_bm25 = 0.4
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        task_type = await QueryUtils().getQuestionType(query_bundle.query_str)
        logger.info(f"current question task_type for retrieval: {task_type}")
        keyword_query_bundle = QueryBundle(query_str=task_type)
        keyword_nodes = self._bm25_retriever.retrieve(keyword_query_bundle)
        retrieve_nodes = self._mix_results(vector_nodes, keyword_nodes)
        return retrieve_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        logger.warning("In CustomMixtureRetriever, under normal circumstances, system cannot reach here!")
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._bm25_retriever.retrieve(query_bundle)
        retrieve_nodes = self._mix_results(vector_nodes, keyword_nodes)
        return retrieve_nodes

    def _mix_results(self, vector_nodes: List[NodeWithScore], keyword_nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        id_dict = {}
        results: List[NodeWithScore] = []
        for node in vector_nodes:
            print("original vector similarity (_id, score): ", node.node_id, node.score)
            if node.node_id not in id_dict:
                node.score = self.weight_faiss * self._distances_to_normalize_similarity(node.score, 1.0)
                results.append(node)
                id_dict[node.node_id] = 1
            else:
                pass  # cannot reach here
        keyword_nodes = self._keyword_bm25_normalize_similarity(keyword_nodes)
        for node in keyword_nodes:
            print("original task type similarity (_id, score): ", node.node_id, np.max(node.score))
            if node.node_id not in id_dict:
                node.score = self.weight_bm25 * np.max(node.score)
                results.append(node)
                id_dict[node.node_id] = 1
            else:
                self._updateNodeScore(results, node.node_id, node.score, self.weight_bm25)

        mixed_results = sorted(results, key=lambda x: x.score, reverse=True)
        for node in mixed_results:
            print("final similarity (_id, score): ", node.node_id, node.score)
        return mixed_results

    def _updateNodeScore(self, results: List[NodeWithScore], doc_id: str, score: float, weight: float) -> List[NodeWithScore]:
        for result in results:
            if result.node_id == doc_id:
                result.score += weight * score
        return results

    def _distances_to_normalize_similarity(self, dis, d_max):
        """
        faiss_distance 归一化
        @Parameters
            dis: int -- faiss_distance
            d_max: int -- 最大距离
        @Returns
            similarity: float -- 归一化后的相似度得分
        """
        dis = np.asarray(dis)
        # 将无穷大的距离替换为 d_max，表示完全不相似的情况
        dis = np.where(np.isinf(dis), d_max, dis)
        similarity = 1 - (dis / d_max)
        similarity = np.clip(similarity, 0, 1)
        return similarity

    def _keyword_bm25_normalize_similarity(self, keyword_nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        bm25_scores = [node.score for node in keyword_nodes]
        normalized_scores = self.normalize_bm25_scores_with_softmax(bm25_scores)
        if len(keyword_nodes) != len(normalized_scores):
            raise ValueError("Length of keyword_nodes and normalized_scores must be the same")
        for node, score in zip(keyword_nodes, normalized_scores):
            node.score = score
        return keyword_nodes

    def normalize_bm25_scores_with_min_max(self, scores):
        """
        对 BM25分数进行归一化
        @Parameters
            scores: array_like -- BM25 原始得分数组。

        @Returns
            normalized_scores: array_like -- 归一化后的得分数组，范围在 [0, 1] 之间。
        """
        # note 我们这里只检索了两个，不适合用min-max归一化，因此暂时弃用
        logger.warning("Warning: Min-max normalization is not suitable for us, cause we only retrieve two nodes.")
        scores = np.asarray(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        # 如果最大值和最小值相同，所有分数都归一化为 0 或 1
        if max_score == min_score:
            return np.zeros_like(scores)  # 所有得分相同的情况，归一化为 0

        # Min-Max normalization
        normalized_scores = (scores - min_score) / (max_score - min_score)
        return normalized_scores

    def normalize_bm25_scores_with_softmax(self, scores: list[float]) -> list[float]:
        exp_scores = [math.exp(score) for score in scores]
        sum_exp_scores = sum(exp_scores)
        normalized_scores = [exp_score / sum_exp_scores for exp_score in exp_scores]
        return normalized_scores

    def _mix_result_with_and_or(self, vector_nodes: List[NodeWithScore], keyword_nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        # note: 此方法根据 mode参数 实现了检索结果混合的 与或逻辑
        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}
        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})
        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes
