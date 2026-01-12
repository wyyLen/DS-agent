import json
import time
import warnings
from typing import Optional, List

import json5
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic.v1 import validator

from metagpt.actions import ExecuteNbCode
from metagpt.actions.ds_agent.conclude_res import Conclusion
from metagpt.actions.lats.lats_react import GenerateAction, ExecuteAction, fix_json
from metagpt.config2 import Config
from metagpt.const import METAGPT_ROOT
from metagpt.logs import logger
from metagpt.rag.engines.customSolutionSamplesGenerate import SolutionSpaceGenerateEngine
from metagpt.schema import Task, Plan, Message
from metagpt.utils.common import CodeParser


# desc 节点设计上暂时保持了与LATS相似的结构，区别主要在于：
#   1. action 在这里代表生成的 ds代码
#   2. execute_code 模块用来记录每一个节点代码生成和执行的情况，每个节点都有一个完整的notebook
class Node:
    def __init__(self, state, question, parent=None, execute_code: ExecuteNbCode = None, is_success=False):
        self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
        self.parent: Node = parent
        self.question = question
        self.children: list[Node] = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_success = is_success
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False  # desc 所有子节点都探索完全
        self.em = 0  # Exact match, evaluation metric
        self.execute_code = execute_code or ExecuteNbCode()  # desc jupyter代码执行
        self.generate_action = GenerateAction()

    def uct(self):
        alpha_unvisited, alpha_explore = 0.8, 1.4
        if self.visits == 0:
            n_visits = alpha_unvisited
        else:
            n_visits = self.visits
        return self.value / n_visits + alpha_explore * np.sqrt(2 * np.log(self.parent.visits) / n_visits)

    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, reward={self.reward:.2f}, visits={self.visits}, thought={self.state['thought']}, action={self.state['action']}, observation={self.state['observation']})"

    def generate_unique_key(self):
        state_str = str(self.state)
        key = f"depth_{self.depth}_{state_str}"
        return key

    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }


def collect_leaf_nodes(root) -> list[Node]:
    leaf_nodes = []

    def _traverse(node):
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                _traverse(child)
    _traverse(root)

    return leaf_nodes


def collect_trajectory(node):
    trajectory = []
    while node:
        trajectory.append(str(node))
        node = node.parent
    return '\n'.join(reversed(trajectory))


def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['thought']:
            new_segment.append(f"Thought {node.depth}: {node.state['thought']}")
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation']:
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n'.join(reversed(trajectory))


def generate_short_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation']:  # Exclude the observation from the root node
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n'.join(reversed(trajectory))


def trajectory2plan(node: Node) -> Plan:
    task_list = []
    goal = node.question
    while node:    # 由于root节点代表空方案的状态，不包含任何内容，因此这里不能包含root节点（已于2.20调整，新版本初始节点不为空）
        thought = node.state['thought']
        thought_json = thought if isinstance(thought, dict) else json.loads(thought)
        new_task = Task(task_id="1", dependent_task_ids=[],
                        instruction=thought_json['thought'],
                        task_type=thought_json['task_type']
                        )
        task_list.append(new_task)
        node = node.parent
    task_list.reverse()
    for i, task in enumerate(task_list):
        task.task_id = str(i + 1)
        if i > 0:
            task.dependent_task_ids.append(str(i))
    cur_plan = Plan(goal=goal, tasks=task_list)
    return cur_plan


async def get_values(node: Node, goal, explore_trajectories, is_terminals, n_evaluate_sample, reflection_map: map = {}, failed_trajectories=[]):
    values = []
    local_value_cache = {}
    for trajectory, is_terminal in zip(explore_trajectories, is_terminals):  # each partial output
        if trajectory in local_value_cache:  # avoid duplicate candidates
            value = local_value_cache.get(trajectory)
        elif not node.is_success:
            value = 0
        else:
            if not is_terminal:
                value = await node.generate_action.evaluate_current_trajectory(goal, explore_trajectories, n_evaluate_sample, depth=node.depth + 1)
            else:
                value = await node.generate_action.evaluate_terminal_trajectory(goal, explore_trajectories, n_evaluate_sample,
                                                                               reflection_map, failed_trajectories)
            local_value_cache[trajectory] = value
        values.append(value)
    return values


async def evaluate_sub_node(node: Node, goal, n_evaluate_sample, reflection_map: map = {}, failed_trajectories=[]):
    # desc 对 expand 的子节点进行评估

    # child_prompts = [generate_prompt(child) for child in node.children]
    child_prompts, is_terminals = zip(*[(generate_prompt(child), child.is_terminal) for child in node.children])

    votes = await get_values(node, goal, child_prompts, is_terminals, n_evaluate_sample,
                             reflection_map=reflection_map, failed_trajectories=failed_trajectories)

    print(f"Length of votes: {len(votes)}, votes: {votes}, Length of node.children: {len(node.children)}")

    votes = votes + [0] * (len(node.children) - len(votes))  # 容错设计
    for i, child in enumerate(node.children):
        child.value = votes[i]
        child.reward = votes[i]

    return votes


def backpropagate(node: Node, value):
    while node:
        node.visits += 1
        if node.is_terminal:
            tree_search_max_depth = 10
            if node.reward < 5:
                # 动态惩罚系数: 根据深度调整惩罚力度（深度越大惩罚越重）
                penalty = -1 * (node.depth / tree_search_max_depth)
                node.value = (node.value * (node.visits - 1) + penalty) / node.visits
                logger.info(f"Backpropagating with low reward at depth {node.depth}. New value: {node.value}.")
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits
                logger.info(f"Backpropagating with higher reward at depth {node.depth}. New value: {node.value}.")
        else:
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logger.info(f"Backpropagating at depth {node.depth}. New value: {node.value}.")

        if not node.is_success:
            node.value = 0

        node = node.parent


def collect_all_nodes(node):
    """Recursively collect all nodes starting from the given node."""
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_all_nodes(child))
    return nodes


def set_node_score(node: Node, score):
    node.value = score
    node.reward = score


class LanguageAgentTreeSearch(BaseModel):
    # note 如果 LanguageAgentTreeSearch 类不需要 pydantic 提供的数据验证和其他功能，可以考虑不继承自 BaseModel。
    model_config = ConfigDict(arbitrary_types_allowed=True)
    goal: str = ""
    root: Optional[Node] = None
    all_nodes: List[Node] = Field(default_factory=list)
    failed_trajectories: List = Field(default_factory=list)
    terminal_nodes: List = Field(default_factory=list)
    reflection_map: dict = {}
    max_reflections_per_node: int = 3
    solution_space_generate_engine: SolutionSpaceGenerateEngine = SolutionSpaceGenerateEngine()
    use_exp_driven_search: bool = True
    use_dual_reflection: bool = True

    async def enhance_run(self, iterations=10, n_generate_sample=3) -> tuple[str, Node]:
        best_child, all_nodes = await self.run(iterations, n_generate_sample)

        for node in all_nodes:
            await node.execute_code.terminate()

        def summarize_trajectory(node: Node) -> tuple[list[str], list[str]]:
            thoughts, observations = [], []
            current_node = node
            while current_node:
                thoughts.append(current_node.state.get('thought', ''))
                observations.append(current_node.state.get('observation', ''))
                current_node = current_node.parent
            return thoughts[::-1], observations[::-1]

        goal = best_child.question
        thoughts, observations = summarize_trajectory(best_child)
        tasks_with_res = [
            {"task_instruction": t, "task_res": o}
            for t, o in zip(thoughts, observations)
        ]
        rsp = await Conclusion().run(final_goal=goal, tasks_res=tasks_with_res)
        return rsp, best_child

    async def run(self, iterations=10, n_generate_sample=2) -> tuple[Node, List[Node]]:
        self.root = Node(None, self.goal, execute_code=ExecuteNbCode())
        for i in range(iterations):
            logger.info(f"Iteration {i + 1}...")

            # desc: 首次迭代强制进行预分析
            if i == 0:
                await self.force_pda(self.root, self.goal)
                set_node_score(self.root, 7)
                continue

            # desc: 检查所有叶子节点，若有终止节点分值较高，可直接返回
            leaf_nodes = collect_leaf_nodes(self.root)
            leaf_terminal_nodes = [leaf for leaf in leaf_nodes if leaf.is_terminal]
            leaf_terminal_nodes.sort(key=lambda leaf: leaf.reward, reverse=True)
            logger.info(f"Leaf terminal nodes count: {len(leaf_terminal_nodes)}")
            for leaf in leaf_terminal_nodes:
                logger.info(f"Leaf terminal node reward: {leaf.reward}")
            if leaf_terminal_nodes and leaf_terminal_nodes[0].reward >= 7:
                logger.info(f"Terminal node with high reward found at iteration {i + 1}")
                return leaf_terminal_nodes[0], leaf_nodes

            # desc: 节点选择
            max_retries = 5  # 新增重试限制
            retries = 0
            node = self.select_node(self.root)
            while node is None or (node.is_terminal and node.reward < 7):
                logger.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
                node = self.select_node(self.root)
                retries += 1
                if retries >= max_retries:
                    logger.warning("Max selection retries reached, aborting search.")
                    best_child = max(collect_all_nodes(self.root), key=lambda x: x.reward, default=self.root)
                    return best_child, self.all_nodes
            if node is None or node.value == 0:
                logger.info("All paths lead to terminal nodes with a low reward. Ending search.")
                best_child = max(collect_all_nodes(self.root), key=lambda x: x.reward, default=self.root)
                return best_child, self.all_nodes

            if node.is_terminal and node.reward >= 7:
                logger.info(f"Terminal node with high reward found at iteration {i + 1}")
                return node, self.all_nodes

            # desc: 节点扩展
            await self.expand_node(node, self.goal, n_generate_sample=n_generate_sample)
            retries = 0
            while node.is_terminal or not node.children:
                retries += 1
                if retries >= max_retries:
                    logger.warning("Max expansion retries reached, aborting search.")
                    return node, self.all_nodes
                logger.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
                node = self.select_node(self.root)
                await self.expand_node(node, self.goal, n_generate_sample=n_generate_sample)

            best_child_reward, terminal_node = await self.rollout(max(node.children, key=lambda child: child.value),
                                                                  n_generate_sample=n_generate_sample,
                                                                  task=self.root.question, max_depth=10)
            print(f" ---------------------------rollout ended --------------------------------\n")

            self.terminal_nodes.append(terminal_node)
            if terminal_node.reward >= 7:
                logger.info("SUCCESSFUL TRAJECTORY FOUND DURING SIMULATION")
                return terminal_node, []

            backpropagate(terminal_node, best_child_reward)
            all_nodes = [(node, node.value) for node in collect_all_nodes(self.root)]
            self.all_nodes = [node for node, val in all_nodes]
            terminal_nodes_with_high_reward = [node for node in collect_all_nodes(self.root) if node.is_terminal and node.reward >= 7]
            if terminal_nodes_with_high_reward:
                logger.info(f"Terminal node with high reward found at iteration {i + 1}")
                best_node = max(terminal_nodes_with_high_reward, key=lambda x: x.value)
                return best_node, self.all_nodes

            # desc: 当一次模拟没有找到高价值的解，对主路径回溯更新。并反思当前路径
            if self.use_dual_reflection:
                reflection = await node.generate_action.reflect_failed_trajectory(terminal_node)
                self.reflection_map[node.generate_unique_key()] = reflection

            for j, (node, value) in enumerate(all_nodes):
                logger.info(f"Node {1}: {str(node)}")
            logger.info(f"State of all_nodes after iteration {i + 1}: {all_nodes}")

        all_nodes_list = collect_all_nodes(self.root)
        all_nodes_list.extend(self.terminal_nodes)
        logger.info(f"State of all_nodes after all iterations: {all_nodes_list}")
        best_child = max(all_nodes_list, key=lambda x: x.reward)
        if best_child.reward >= 7:
            logger.success("Successful trajectory found")
        else:
            logger.warning("No successful trajectory found")
        if best_child is None:
            best_child = self.root
        return best_child, self.all_nodes

    async def force_pda(self, node, goal):
        instruction, code = await node.generate_action.generate_pda(goal)
        predefined_thought = {
            "task_type": "pda",
            "thought": instruction
        }
        obs, r, is_success, info = await ExecuteAction().step(goal, instruction, code, node.execute_code)
        node.state['thought'] = predefined_thought
        node.state['action'] = code
        node.state['observation'] = obs
        node.is_success = is_success

    def select_node(self, node):
        current_depth = 0
        max_depth = 50  # 防止无限递归

        while node and node.children and current_depth < max_depth:
            current_depth += 1
            terminal_children = [child for child in node.children if child.is_terminal]

            high_reward_terminal_node = next((c for c in terminal_children if c.reward >= 7), None)
            if high_reward_terminal_node:
                return high_reward_terminal_node

            # 剪枝逻辑优化：仅当所有子节点都是低奖励终端节点时才剪枝
            if len(terminal_children) == len(node.children):
                if all(c.reward < 5 for c in terminal_children):
                    if node.parent:
                        logger.debug(f"Pruning node {node} from parent")
                        node.parent.children.remove(node)
                    node = node.parent
                    continue
                else:
                    return max((c for c in terminal_children), key=lambda x: x.reward)

            non_terminal_children = [c for c in node.children if not c.is_terminal]
            if not non_terminal_children:
                continue  # 触发剪枝逻辑
            node = max(non_terminal_children, key=lambda child: child.uct())

        # 返回最后有效节点或根节点作为保底
        return node if node else self.root

    async def expand_node(self, node, goal, n_generate_sample):
        if node.depth >= 10:
            logger.info("Depth limit reached")
            node.is_terminal = True
            return
        new_nodes = await self.generate_new_states(node, goal, n_generate_sample)
        node.children.extend(new_nodes)
        rewards = await evaluate_sub_node(node, goal, n_evaluate_sample=1)
        node.children.sort(key=lambda c: c.reward, reverse=True)
        print(f"expand_node方法中直接返回的奖励分数: {rewards}")
        print(f"expand_node方法中当前节点所有子节点的分值列表: {[child.reward for child in node.children]}")
        for child, reward in zip(node.children, rewards):
            child.reward = reward
        for child in node.children:
            if child.is_terminal and child.reward < 5:
                trajectory = collect_trajectory(child)
                self.failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{child.state['thought']}"})

    async def generate_new_states(self, node: Node, goal, n_generate_sample):
        # desc 得到当前节点所代表路径的整体过程
        cur_trajectory = generate_prompt(node)
        cur_short_trajectory = generate_short_prompt(node)
        workflow_exps = []
        if node.depth >= 2 and self.use_exp_driven_search:
            logger.info(f"-------------- 准备检索工作流经验 ---------------")
            cur_plan = trajectory2plan(node)
            workflow_exps = self.solution_space_generate_engine.run(cur_plan)
            logger.info(f"相关工作流经验数量: {len(workflow_exps)}, 当前工作流: {cur_plan}")
        # desc 生成动作采样空间
        sampled_actions = await node.generate_action.generate_solution_space(cur_short_trajectory, f"\nThought {node.depth + 1}: ", n_generate_sample, workflow_exps,
                                                                             reflection_map=self.reflection_map, failed_trajectories=self.failed_trajectories)
        logger.info(f"Sampled num: {len(sampled_actions)}, SAMPLED ACTIONS: {sampled_actions}")

        tried_actions = []
        unique_states = {}  # Store unique states here
        layer_reflections = []
        for i, item in enumerate(sampled_actions):
            thought_dict = item["thought"]
            action = item["response"]
            new_state = node.state.copy()
            new_exec = await node.execute_code.model_copy(deep=True)

            print(f"type of thought_dict: {type(thought_dict)}, content: {thought_dict}")
            original_thought = thought_dict if isinstance(thought_dict, dict) else json5.loads(thought_dict)
            original_thought['task_type'] = original_thought.get('task_type', 'other')
            original_thought['thought'] = original_thought.get('thought', '')
            original_unique_key = f"{original_thought['task_type']}::{original_thought['thought']}"
            if original_unique_key in unique_states:
                continue

            if original_thought['task_type'] == 'finish':
                logger.success("Goal achieved")

                def collect_all_result(cur_node: Node) -> str:
                    msg_list = []
                    while cur_node:
                        cur_depth = cur_node.depth
                        msg_list.append(f"observation {cur_depth}: {cur_node.state['observation']}\n")
                        msg_list.append(f"action {cur_depth}: {cur_node.state['action']}\n")
                        cur_node = cur_node.parent
                    msg_list.reverse()
                    all_observation = "".join(msg_list)
                    return all_observation

                new_state['thought'] = original_thought
                new_state['action'] = ""
                new_state['observation'] = "\n all actions and observations:" + collect_all_result(node)
                new_node = Node(state=new_state, question=node.question, parent=node, execute_code=new_exec)
                new_node.is_terminal = True
                new_node.depth = node.depth + 1
                unique_states[original_unique_key] = new_node
                continue

            action_code = CodeParser.parse_code(block=None, text=action, lang='python')
            tried_actions.append(action_code)

            if not action_code:
                continue

            current_thought, task_type = original_thought, original_thought['task_type']
            current_action_code = action_code
            current_obs, current_r, current_is_success, current_info = await ExecuteAction().step(goal, current_thought, current_action_code, new_exec)
            reflection_count = 0
            reflection_attempt = [{"action": current_action_code, "observation": current_obs}]

            while not current_is_success and reflection_count < self.max_reflections_per_node and node.is_success:
                reflection_count += 1
                logger.info(f"code execution error, start reflection in {reflection_count} counts.")
                error_msg = current_obs if current_obs else "unknown error"
                reflection_thought, current_action_code = await node.generate_action.debug_with_reflection(node, current_thought, current_action_code, error_msg, layer_reflections)
                current_thought = {
                    "thought": reflection_thought,
                    "task_type": task_type
                }
                current_obs, current_r, current_is_success, current_info = await ExecuteAction().step(goal, current_thought, current_action_code, new_exec)
                reflection_attempt.append({"action": current_action_code, "observation": current_obs})
                if current_is_success:
                    logger.info(f"reflection success in {reflection_count} rounds")
                    break

            # 当前节点多次反思仍失败，
            # 且父节点成功，（若不成功说明问题不在这一节点）
            # 且当前不是最后一个节点（最后一个节点没有层级反思的必要），进行层级反思。
            if not current_is_success and reflection_count == self.max_reflections_per_node and node.is_success and i < len(sampled_actions) - 1 and self.use_dual_reflection:
                layer_reflection = await node.generate_action.reflect_failed_node(reflection_attempt)
                layer_reflections.append(layer_reflection)

            # desc 无论最终是否成功，只要没采样相同路径，都需要创建节点
            if original_unique_key in unique_states:
                await new_exec.terminate()
                continue

            new_state['thought'] = current_thought
            new_state['action'] = current_action_code
            new_state['observation'] = current_obs
            new_node = Node(state=new_state, question=node.question, parent=node, execute_code=new_exec,
                            is_success=current_is_success)
            new_node.depth = node.depth + 1
            new_node.is_terminal = (current_thought['task_type'] == 'finish')
            unique_states[original_unique_key] = new_node

            if new_node.is_terminal and new_node.reward < 5:
                trajectory = collect_trajectory(new_node)
                self.failed_trajectories.append({
                    'trajectory': trajectory,
                    'final_answer': f"{current_action_code}\n{current_obs}"
                })

        return list(unique_states.values())

    async def rollout(self, node: Node, task, n_generate_sample, max_depth=8):
        logger.info("ROLLING OUT")
        depth = node.depth
        rewards, tmp_node = [], node  # 记录当前路径的分数
        while tmp_node:
            rewards.append(tmp_node.reward)
            tmp_node = tmp_node.parent

        while not node.is_terminal and depth < max_depth and node.is_success:
            logger.info(f"ROLLING OUT {depth}")
            # 节点扩展
            await self.expand_node(node, self.goal, n_generate_sample=n_generate_sample)

            # desc: 若最优终止节点的reward大于7，直接返回并触发旁路 backpropagate
            #  由于expand方法中已经进行了排序，因此只需要判断第一个即可
            for child in node.children:
                if child.is_terminal and child.reward >= 7:
                    for sibling in node.children:
                        if sibling != child:
                            backpropagate(sibling, (sum(rewards) + sibling.reward) / (len(rewards) + 1))
                    rewards.append(child.reward)
                    return sum(rewards) / len(rewards), child

            rewards = [child.reward for child in node.children]
            print(f"当前节点所有子节点的分值列表: {rewards}")
            best_child = max(node.children, key=lambda children: children.reward)

            # desc: 非终止节点的旁路引发的 backpropagate
            for sibling in node.children:
                if sibling != best_child:
                    backpropagate(sibling, (sum(rewards) + sibling.reward) / (len(rewards) + 1))

            rewards.append(best_child.reward)
            node = best_child
            # print(f"新节点价值:{node.reward}, 新节点is_terminal:{node.is_terminal}, 新节点状态: {node.state}")
            depth += 1

        if node.is_terminal:
            # 由于在expand_node 中已经进行了节点reward和value的更新, 这里的赋值反而会造成数据污染
            # node.reward = node.parent.reward
            # node.value = node.reward
            rewards.append(node.reward)

        if depth == max_depth:
            logger.info(f"reaching max_depth, rollout ended")
            rewards.append(0)

        if not node.is_success:
            logger.warning(f"node error, rollout ended")
            rewards.append(0)

        logger.info("ROLLOUT FINISHED")
        return sum(rewards) / len(rewards), node

    def calculate_total_cost(self) -> tuple[int, int]:
        # desc: 从 root 开始遍历所有节点，统计所有 Action 的 LLM 调用总开销
        #     lats的开销统计有特殊性，除了根节点外，所有的节点开销都统计于其父节点上（因为是由其父节点扩展的)
        prompt_tokens, completion_tokens = 0, 0

        def _traverse(node: Node):
            nonlocal prompt_tokens, completion_tokens
            if not node:
                return

            if node.generate_action:
                prompt_token = node.generate_action.llm.cost_manager.total_prompt_tokens
                completion_token = node.generate_action.llm.cost_manager.total_completion_tokens
                prompt_tokens += prompt_token
                completion_tokens += completion_token

            for child in node.children:
                _traverse(child)

        if self.root:
            _traverse(self.root)

        return prompt_tokens, completion_tokens
