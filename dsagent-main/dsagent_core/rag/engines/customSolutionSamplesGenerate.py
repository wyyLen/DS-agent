import json
import threading
import time
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Optional, Union

import numpy as np
import requests

from dsagent_core.actions.ds_agent.fixed_plan_for_test import get_fixed_plan
from dsagent_core.const import EXAMPLE_DATA_PATH
from dsagent_core.logs import logger
from dsagent_core.rag.engines.GraphMatching.graph import GraphSet
from dsagent_core.rag.engines.customEmbeddingComparisonEngine import CustomEmbeddingComparisonEngine
from dsagent_core.rag.engines.graphUtils import _getGraphEditDistance, _json2plan
from metagpt.schema import Plan, Task
from concurrent.futures import ThreadPoolExecutor, as_completed


def vf2_isomorphism(sub_graph: GraphSet, target_graph: GraphSet):
    embedding_engine = CustomEmbeddingComparisonEngine()
    local_embedding_config = dict(
        base_url="your_local_embedding_url",
        model_name="nomic-embed-text:latest"
    )
    openai_embedding_config = dict(
        base_url="your_openai_embedding_url",
        model_name="text-embedding-3-small",
        api_key="your_api_key"
    )
    server_embedding_config = dict(
        base_url="your_server_embedding_url",
        model_name="nomic-embed-text:latest"
    )
    # 本方法的目标就是去计算源图与目标图的相似性。
    # desc: VF2 algorithm requires
    #   1. 对于小图中每个节点，大图中都要有一个对应的节点与之对应，并且这样一对一对的节点构成了集合 mapping
    #   2. 小图中每条边，大图中都有一条边与之对应，并且他们两端的节点一一对应
    #   3. 每对对应节点的label要相同，也就是这俩节点类型或属性相同
    #   4. 每对对应边的label要相同，也就是说这俩边的类型或属性相同
    #   5. （可选）小图中任意两个节点，如果他们对应的大图中的节点之间有一条边，那么小图中这两个节点之间也得有条边
    # func: our implementation of VF2 algorithm
    #   1. 遵循原始算法要求，我们保持小图和大图的节点一一对应，并构成集合 mapping
    #   2. 遵循原始算法要求，我们保持小图的任务依赖关系在大图中均存在
    #   3. 遵循原始算法要求，我们要求小图中的节点任务类型与大图中的节点任务类型一致
    #   4. 与原始算法不同，我们的边并没有属性或类型，因此无需考虑
    #   5. 与原始算法不同，大图的边在子图中不必要，因此不做考虑
    def is_feasible(mapping: dict, node1: int, node2: int) -> bool:
        # debug
        # print(f"sub_graph.curVSet(0)[node1]: {sub_graph.curVSet(0)[str(node1 + 1)]}")
        # desc: 1.检查任务类型是否一致
        if sub_graph.curVSet(0)[str(node1 + 1)]['task_type'] != target_graph.curVSet(0)[str(node2 + 1)]['task_type']:
            return False
        # 检查 node2 是否已经被映射
        if node2 in mapping.values():
            return False
        # 检查边的一致性
        for edge in sub_graph.neighbor(0, node1):
            v1, v2 = edge.split(":")
            neighbor = int(v2) - 1 if int(v1) - 1 == node1 else int(v1) - 1
            if neighbor in mapping:
                mapped_neighbor = mapping[neighbor]
                if f"{node2 + 1}:{mapped_neighbor + 1}" not in target_graph.curESet(0) and f"{mapped_neighbor + 1}:{node2 + 1}" not in target_graph.curESet(0):
                    return False
        # 检查小图中节点之间的边是否在大图中存在
        for node in mapping:
            if f"{node1 + 1}:{node + 1}" in sub_graph.curESet(0) or f"{node + 1}:{node1 + 1}" in sub_graph.curESet(0):
                if f"{node2 + 1}:{mapping[node] + 1}" not in target_graph.curESet(
                        0) and f"{mapping[node] + 1}:{node2 + 1}" not in target_graph.curESet(0):
                    return False
        return True

    def get_embedding_from_mapping(mapping: dict) -> float:
        sub_graph_instruction, target_graph_instruction = "", ""
        for k, v in mapping.items():
            sub_graph_instruction += sub_graph.curVSet(0)[str(k + 1)]["instruction"]
            target_graph_instruction += target_graph.curVSet(0)[str(v + 1)]["instruction"]
        embedding_similarity = embedding_engine.run(
            sub_graph_instruction,
            target_graph_instruction,
            embedding_config=local_embedding_config
        )
        return embedding_similarity

    def search(mapping: dict) -> tuple[bool, float | None]:
        if len(mapping) == len(sub_graph.curVSet(0)):
            return True, get_embedding_from_mapping(mapping)
        unmapped_nodes1 = [i for i in range(len(sub_graph.curVSet(0))) if i not in mapping]
        unmapped_nodes2 = [i for i in range(len(target_graph.curVSet(0))) if i not in mapping.values()]
        # print(f"unmapped_nodes1: {unmapped_nodes1}, unmapped_nodes2: {unmapped_nodes2}")
        # desc g1作为源图，尝试映射到g2。
        for node1 in unmapped_nodes1:
            for node2 in unmapped_nodes2:
                if is_feasible(mapping, node1, node2):
                    mapping[node1] = node2
                    if search(mapping):
                        return True, get_embedding_from_mapping(mapping)
                    del mapping[node1]
        return False, None

    return search({})


def flatten_workflow_v3_optimized(workflow: Plan, max_sequences: Optional[int] = None) -> List[Plan]:
    """优化后的扁平化方法，支持生成数量限制和分支连续性保证
    Args:
        workflow: 包含目标任务和任务列表的工作流对象
        max_sequences: 最大生成序列数，None表示自动计算
    Returns:
        List[Plan]: 合法执行序列列表（数量不超过max_sequences）
    """
    task_list = workflow.tasks
    goal = workflow.goal
    task_map = {task.task_id: task for task in task_list}
    task_ids = [task.task_id for task in task_list]
    n = len(task_list)

    # 动态计算最大序列数
    if max_sequences is None:
        def heuristic_max_sequences(n: int) -> int:
            """根据任务数量启发式确定最大生成序列数"""
            if n <= 5:
                return 2  # 小规模任务覆盖全排列
            elif n <= 10:
                return 3  # 中等规模任务
            else:
                return 3  # 大规模任务限制生成数量
        max_sequences = heuristic_max_sequences(n)

    # 构建依赖关系图
    in_degree: Dict[str, int] = {}
    successors: Dict[str, List[str]] = defaultdict(list)
    predecessors: Dict[str, List[str]] = defaultdict(list)
    for task in task_list:
        in_degree[task.task_id] = len(task.dependent_task_ids)
        for dep_id in task.dependent_task_ids:
            successors[dep_id].append(task.task_id)
            predecessors[task.task_id].append(dep_id)

    all_sequences: List[List[str]] = []
    sequence_count = 0  # 生成序列计数器

    def backtrack(current_path: List[str], current_degree: Dict[str, int]):
        nonlocal sequence_count
        if sequence_count >= max_sequences:
            return

        # 强制任务检测：反向遍历路径寻找首个可展开节点
        forced_tasks = []
        if current_path:
            for task_id in reversed(current_path):
                available_successors = [
                    succ for succ in successors[task_id]
                    if succ not in current_path and
                       current_degree[succ] == 0 and
                       all(pre in current_path for pre in predecessors[succ])
                ]
                if available_successors:
                    forced_tasks = available_successors
                    break

        # 确定可用任务集合
        available = (
            list(dict.fromkeys(forced_tasks)) if forced_tasks else [
                tid for tid in task_ids
                if current_degree[tid] == 0 and tid not in current_path
        ])

        # 终止条件
        if len(current_path) == len(task_ids):
            all_sequences.append(current_path.copy())
            sequence_count += 1
            return

        # 遍历可用任务（优先处理强制任务）
        for task_id in available:
            if sequence_count >= max_sequences:
                break  # 提前终止

            # 复制入度状态以避免污染其他分支
            new_degree = current_degree.copy()
            new_degree[task_id] = -1  # 标记为已处理

            # 更新后续任务入度
            for succ in successors[task_id]:
                if new_degree[succ] > 0:
                    new_degree[succ] -= 1

            # 递归探索
            backtrack(current_path + [task_id], new_degree)
            if sequence_count >= max_sequences:
                break  # 上层递归快速终止

    backtrack([], in_degree.copy())

    # 转换为Plan对象列表
    return [
        Plan(
            goal=goal,
            tasks=[task_map[tid] for tid in sequence]
        ) for sequence in all_sequences
    ]


def flatten_workflow_v2_optimized(workflow: Plan) -> List[Plan]:
    # desc: 优化后的扁平化方法，确保独立分支任务连续执行
    task_list = workflow.tasks
    goal = workflow.goal
    task_map = {task.task_id: task for task in task_list}
    task_ids = [task.task_id for task in task_list]

    # 构建依赖关系图
    in_degree: Dict[str, int] = {}
    successors: Dict[str, List[str]] = defaultdict(list)
    predecessors: Dict[str, List[str]] = defaultdict(list)
    for task in task_list:
        in_degree[task.task_id] = len(task.dependent_task_ids)
        for dep_id in task.dependent_task_ids:
            successors[dep_id].append(task.task_id)
            predecessors[task.task_id].append(dep_id)

    all_sequences: List[List[str]] = []

    def backtrack(current_path: List[str], current_degree: Dict[str, int]):
        # 强制任务检测：反向遍历路径，找到第一个有可用后续的任务
        forced_tasks = []
        if current_path:
            # 反向遍历当前路径中的任务，寻找第一个有可用后续的节点
            for task_id in reversed(current_path):
                available_successors = []
                for successor in successors[task_id]:
                    if (successor not in current_path and
                        current_degree[successor] == 0 and
                        all(pre in current_path for pre in predecessors[successor])):
                        available_successors.append(successor)
                if available_successors:
                    forced_tasks = available_successors
                    break  # 找到后停止遍历

        # 确定可用任务集合
        available = []
        if forced_tasks:
            available = list(dict.fromkeys(forced_tasks))  # 去重保持顺序
        else:
            available = [tid for tid in task_ids
                         if current_degree[tid] == 0
                         and tid not in current_path]

        # 终止条件
        if len(current_path) == len(task_ids):
            all_sequences.append(current_path.copy())
            return

        # 处理分支选择（优先处理强制任务）
        for task_id in available:
            new_degree = current_degree.copy()
            new_degree[task_id] = -1  # 标记为已处理

            # 更新后续任务的入度
            for successor in successors[task_id]:
                if new_degree[successor] > 0:
                    new_degree[successor] -= 1

            # 递归探索
            backtrack(current_path + [task_id], new_degree)

    backtrack([], in_degree.copy())

    # 转换为Plan对象列表
    return [
        Plan(
            goal=goal,
            tasks=[task_map[tid] for tid in sequence]
        ) for sequence in all_sequences
    ]


def flatten_workflow_v1_optimized(workflow: Plan):
    # note: 存在过多冗余工作流，已弃用
    task_list = workflow.tasks
    goal = workflow.goal

    # 构建快速访问结构
    task_map = {task.task_id: task for task in task_list}
    task_ids = [task.task_id for task in task_list]

    # 构建依赖关系图
    in_degree: Dict[str, int] = {}  # 任务剩余未满足的前置依赖计数
    successors: Dict[str, List[str]] = defaultdict(list)  # 后向依赖关系图

    for task in task_list:
        in_degree[task.task_id] = len(task.dependent_task_ids)
        for dep_id in task.dependent_task_ids:
            successors[dep_id].append(task.task_id)

    # 检测循环依赖（Kahn算法）
    temp_degree = in_degree.copy()
    queue = [tid for tid, cnt in temp_degree.items() if cnt == 0]
    processed = 0
    while queue:
        node = queue.pop(0)
        processed += 1
        for successor in successors[node]:
            temp_degree[successor] -= 1
            if temp_degree[successor] == 0:
                queue.append(successor)
    if processed != len(task_ids):
        raise ValueError("工作流存在循环依赖，无法生成合法序列")

    # 回溯法生成所有拓扑排序
    all_sequences: List[List[str]] = []

    def backtrack(current_path: List[str], current_degree: Dict[str, int]):
        # 获取当前可执行任务（入度为0且未在路径中）
        available = [
            tid for tid in task_ids
            if current_degree[tid] == 0 and tid not in current_path
        ]

        # 终止条件：所有任务已处理
        if len(current_path) == len(task_ids):
            all_sequences.append(current_path.copy())
            return

        # 遍历所有可能的分支
        for task_id in available:
            new_degree = current_degree.copy()

            # 标记当前任务为已处理
            new_degree[task_id] = -1

            # 更新后续任务的入度
            for successor in successors[task_id]:
                if new_degree[successor] > 0:
                    new_degree[successor] -= 1

            # 递归探索
            backtrack(current_path + [task_id], new_degree)

    backtrack([], in_degree.copy())

    # 转换为Plan对象列表
    return [
        Plan(
            goal=goal,
            tasks=[task_map[tid] for tid in sequence]
        ) for sequence in all_sequences
    ]


def flatten_workflow(workflow):
    # desc 将层次化的工作流进行扁平化
    task_list = workflow.tasks
    goal = workflow.goal
    task_map = {task.task_id: task for task in task_list}
    dependencies = {task.task_id: set(task.dependent_task_ids) for task in task_list}  # 前向依赖
    export_dependencies = {task.task_id: set() for task in task_list}  # 后向依赖
    for task in task_list:
        for dep in task.dependent_task_ids:
            export_dependencies[dep].add(task.task_id)
    # 找到所有根节点和叶子节点
    root_nodes = [task_id for task_id, deps in dependencies.items() if not deps]
    end_nodes = [task_id for task_id, exports in export_dependencies.items() if not exports]
    all_sequences = []

    def backtrack(current_sequence, current_node):
        # 如果当前节点是叶子节点，则将当前路径添加到结果中
        if current_node in end_nodes:
            sequence_json = [task_map[task_id].copy() for task_id in current_sequence]
            all_sequences.append(sequence_json)
            return
        for next_node in export_dependencies[current_node]:
            # 如果下一个节点已经在当前路径中，则跳过（避免循环依赖）
            if next_node in current_sequence:
                continue
            current_sequence.append(next_node)
            backtrack(current_sequence, next_node)
            current_sequence.pop()

    for root_node in root_nodes:
        backtrack([root_node], root_node)

    results = []
    for seq_json in all_sequences:
        for i, task in enumerate(seq_json, 1):
            task.task_id = str(i)
            task.dependent_task_ids = [str(i - 1)] if i > 1 else []
        results.append(Plan(goal=goal, tasks=seq_json))

    return results


class SolutionSpaceGenerateEngine:
    workflow_bank: dict[GraphSet, str] = {}
    workflow_ged_threshold: int
    _lock = threading.Lock()  # 线程安全锁

    def __init__(self, max_workers: int = 32):
        WORKFLOW_EXP_PATH_CLEAN = EXAMPLE_DATA_PATH / "exp_bank/workflow_exp2_clean.json"
        self.workflow_bank = {}
        workflow_bank_count = 0
        with open(WORKFLOW_EXP_PATH_CLEAN, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in data:
                futures.append(executor.submit(
                    self._process_item,  # 提交单个item处理任务
                    item_data=item
                ))
            for future in as_completed(futures):
                graphs, exp = future.result()
                with self._lock:  # 线程安全写入
                    for graph in graphs:
                        self.workflow_bank[graph] = exp
                        workflow_bank_count += 1

        print(f"current workflow_bank count: {workflow_bank_count}")
        self.workflow_ged_threshold = 10

    def _process_item(self, item_data: dict) -> tuple[list[GraphSet], str]:
        # desc: 初始化方法中，扁平化工作流的单条数据处理方法
        cur_plan = _json2plan(item_data["workflow"])
        workflows = flatten_workflow_v3_optimized(cur_plan)
        return [GraphSet(flow) for flow in workflows], item_data["exp"]

    def run(self, workflow: Plan, min_exp_num: int = 2, max_exp_num: int = 5):
        cur_trajectory = GraphSet(workflow)
        candidates = []
        logger.info(f"cur_trajectory: {cur_trajectory.graphSet()}")
        stop_event = threading.Event()  # 控制任务终止的信号量

        def process_graph(graph):
            # desc: VF2检索的单条数据处理方法 with 提前终止检查
            if stop_event.is_set():
                return None  # 立即终止当前任务
            vf2_success, similarity = vf2_isomorphism(cur_trajectory, graph)
            if vf2_success:
                return {"graph": graph, "score": similarity}
            return None

        with ThreadPoolExecutor() as executor:
            # 提交所有任务并维护future映射
            futures = {executor.submit(process_graph, graph): graph for graph in self.workflow_bank.keys()}
            high_score_count = 0  # 高分项计数器
            try:
                # 实时处理完成的任务
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        candidates.append(result)
                        # 高分项判断与中断逻辑
                        if result["score"] > 0.7:
                            high_score_count += 1
                            if high_score_count >= max_exp_num:
                                logger.info(f"已达到最大示例数 {max_exp_num}，触发提前终止")
                                stop_event.set()  # 设置终止信号
                                # 取消所有未完成任务
                                for f in futures:
                                    if not f.done():
                                        f.cancel()
                                break  # 中断结果收集循环
            except Exception as e:
                logger.error(f"任务处理异常: {str(e)}")

        candidates_count = len(candidates)
        candidates = [candidate for candidate in candidates if candidate["score"] >= 0]
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        # desc: 基于图编辑距离的补充方法 （目前基本上用不到）
        logger.info(f"vf2_candidates: {len(candidates)}")
        all_workflow = list(self.workflow_bank.keys())
        all_workflow = sorted(all_workflow, key=lambda x: _getGraphEditDistance(cur_trajectory, x), reverse=False)
        logger.info(f"all_workflow: {len(all_workflow)}, "
                    f"first: {_getGraphEditDistance(cur_trajectory, all_workflow[0])}, "
                    f"last: {_getGraphEditDistance(cur_trajectory, all_workflow[-1])}. ")
        while candidates_count < min_exp_num:
            for idx, item in enumerate(candidates):
                if _getGraphEditDistance(cur_trajectory, item["graph"]) < self.workflow_ged_threshold:
                    candidates.append({
                        "graph": all_workflow[idx],
                        "score": 0
                    })
                    candidates_count += 1
                if candidates_count >= min_exp_num:
                    break

        results = []
        for candidate in candidates:
            results.append(self.workflow_bank.get(candidate["graph"]))
        return results


def try_with_cases():
    target_workflow = get_fixed_plan(1812)
    cur_trajectory = Plan(goal="", tasks=[
        Task(task_id="1", dependent_task_ids=[],
             instruction="Load and inspect the abalone dataset to understand its structure and the available columns.",
             task_type="pda"),
        Task(task_id="2", dependent_task_ids=["1"],
             instruction="Calculate the Pearson correlation coefficient between the length and the weight of the whole abalone.",
             task_type="correlation analysis")
    ])
    target_graph, sub_graph = GraphSet(target_workflow), GraphSet(cur_trajectory)
    # res = vf2_isomorphism(sub_graph, target_graph)
    # print(res)
    engine = SolutionSpaceGenerateEngine()
    candidates = engine.run(cur_trajectory)
    for i, candidate in enumerate(candidates):
        print(f"candidate {i}: {candidate}")


def try_flatten():
    exist_workflow = get_fixed_plan(1812)
    flattened_sequences = flatten_workflow_v2_optimized(exist_workflow)
    print(f"flatten workflow count: {len(flattened_sequences)}")

    # 打印所有扁平化后的工作流序列
    for idx, sequence in enumerate(flattened_sequences, 1):
        print(f"Sequence {idx}:")
        print(sequence)
        print("\n")


if __name__ == '__main__':
    pass
    # start_time = time.time()
    try_with_cases()
    # # try_flatten()
    # print(f"total time: {time.time() - start_time} seconds")
