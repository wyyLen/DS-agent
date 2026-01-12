from typing import Dict, Set, List, Optional

from metagpt.schema import Plan, Task

from collections import defaultdict
from typing import List, Set, Dict

from typing import List, Dict, Optional
from collections import defaultdict


def flatten_workflow_optimized_v3(
        workflow: Plan,
        max_sequences: Optional[int] = None
) -> List[Plan]:
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



def flatten_workflow_optimized_ds_v2(workflow: Plan) -> List[Plan]:
    """优化后的扁平化方法，确保独立分支任务连续执行"""
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


def flatten_workflow_optimized(workflow: Plan) -> List[Plan]:
    """优化后的扁平化方法，确保独立分支任务连续执行
    Args:
        workflow: 包含目标任务和任务列表的工作流对象
    Returns:
        List[Plan]: 所有可能的合法执行序列，确保分支连续性
    """
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
        # 强制任务检测（递归检测后续链式任务）
        forced_tasks = []
        if current_path:
            last_task = current_path[-1]
            # 深度优先收集所有可立即执行的后续任务
            stack = [(last_task, successors[last_task])]
            while stack:
                current_node, next_nodes = stack.pop()
                for node in next_nodes:
                    if current_degree[node] == 0 and node not in current_path:
                        if all(pre in current_path for pre in predecessors[node]):
                            forced_tasks.append(node)
                            # 继续检查该节点的后续
                            stack.append((node, successors[node]))

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


def flatten_workflow(workflow: Plan):
    """将层次化工作流扁平化为所有可能的合法任务序列

    Args:
        workflow: 包含目标任务和任务列表的工作流对象

    Returns:
        List[Plan]: 所有可能的合法执行序列（拓扑排序结果）
    """
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


if __name__ == '__main__':
    tasks1 = [
        Task(task_id='1', dependent_task_ids=[], instruction='Load and inspect the abalone dataset', task_type='pda'),
        Task(task_id='2', dependent_task_ids=['1'], instruction='Calculate the Pearson correlation coefficient',
             task_type='correlation analysis'),
        Task(task_id='3', dependent_task_ids=['1'], instruction="Create a new feature 'volume'",
             task_type='feature engineering'),
        Task(task_id='4', dependent_task_ids=['3'], instruction='Split the dataset', task_type='data preprocessing'),
        Task(task_id='5', dependent_task_ids=['4'], instruction='Train a linear regression model (original features)',
             task_type='machine learning'),
        Task(task_id='6', dependent_task_ids=['4'], instruction="Train a linear regression model (with 'volume')",
             task_type='machine learning'),
        Task(task_id='7', dependent_task_ids=['5', '6'], instruction='Calculate RMSE', task_type='machine learning')
    ]

    tasks2 = [
        Task(task_id='1', dependent_task_ids=[],
             instruction='List all files in the input directory to check available datasets.', task_type='pda', code='',
             result='', is_success=False, is_finished=False),
        Task(task_id='2', dependent_task_ids=['1'], instruction='Load the train.csv and store.csv datasets.',
             task_type='data preprocessing', code='', result='', is_success=False, is_finished=False),
        Task(task_id='3', dependent_task_ids=['2'],
             instruction="Merge the train and store datasets on the 'Store' column.", task_type='data preprocessing',
             code='', result='', is_success=False, is_finished=False),
        Task(task_id='4', dependent_task_ids=['3'], instruction='Check for null values in the merged dataset.',
             task_type='data preprocessing', code='', result='', is_success=False, is_finished=False),
        Task(task_id='5', dependent_task_ids=['4'],
             instruction="Fill missing values for 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', and 'CompetitionDistance' with zeros.",
             task_type='data preprocessing', code='', result='', is_success=False, is_finished=False),
        Task(task_id='6', dependent_task_ids=['5'],
             instruction="For rows where 'CompetitionDistance' is filled with zero, fill 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' with zero.",
             task_type='data preprocessing', code='', result='', is_success=False, is_finished=False),
        Task(task_id='7', dependent_task_ids=['6'],
             instruction="Fill remaining missing values for 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' with their respective modes.",
             task_type='data preprocessing', code='', result='', is_success=False, is_finished=False),
        Task(task_id='8', dependent_task_ids=['7'],
             instruction="Drop the columns 'StateHoliday', 'StoreType', 'Assortment', and 'PromoInterval' from the dataset.",
             task_type='feature engineering', code='', result='', is_success=False, is_finished=False),
        Task(task_id='9', dependent_task_ids=['8'],
             instruction="Convert the 'Date' column to datetime format and set it as the index of the dataframe.",
             task_type='data preprocessing', code='', result='', is_success=False, is_finished=False),
        Task(task_id='10', dependent_task_ids=['9'],
             instruction='Plot the time series of sales to visualize trends and seasonality.',
             task_type='distribution analysis', code='', result='', is_success=False, is_finished=False),
        Task(task_id='11', dependent_task_ids=['9'],
             instruction='Split the data into training and testing sets using TimeSeriesSplit.',
             task_type='data preprocessing', code='', result='', is_success=False, is_finished=False),
        Task(task_id='12', dependent_task_ids=['11'],
             instruction='Scale the features and target variable using MinMaxScaler.', task_type='data preprocessing',
             code='', result='', is_success=False, is_finished=False),
        Task(task_id='13', dependent_task_ids=['12'], instruction='Build and compile a simple neural network model for regression.',
             task_type='machine learning-Linear Regression', code='', result='', is_success=False, is_finished=False),
        Task(task_id='14', dependent_task_ids=['13'],
             instruction='Train the neural network model on a subset of the training data and evaluate using the R2 score.',
             task_type='machine learning-Decision Tree', code='', result='', is_success=False, is_finished=False),
        Task(task_id='15', dependent_task_ids=['14'],
             instruction='Reshape the data for LSTM input and adjust the training and testing sets accordingly.',
             task_type='data preprocessing', code='', result='', is_success=False, is_finished=False),
        Task(task_id='16', dependent_task_ids=['15'], instruction='Build and compile an LSTM model for regression.',
             task_type='machine learning-Linear Regression', code='', result='', is_success=False, is_finished=False),
        Task(task_id='17', dependent_task_ids=['16'],
             instruction='Train the LSTM model and evaluate using the R2 score.', task_type='machine learning', code='',
             result='', is_success=False, is_finished=False),
        Task(task_id='18', dependent_task_ids=['14', '17'], instruction='Compare the test mean squared error of the neural network and LSTM models.',
             task_type='machine learning', code='', result='',
             is_success=False, is_finished=False),
        Task(task_id='19', dependent_task_ids=['18'],
             instruction='Use the trained models to forecast sales and plot the predictions against actual sales.',
             task_type='machine learning-Linear Regression', code='', result='', is_success=False, is_finished=False),
        Task(task_id='20', dependent_task_ids=['19'],
             instruction="Prepare the final predictions for submission by creating a DataFrame with 'Id' and 'Sales' columns and Output the result with print() function.",
             task_type='data preprocessing', code='', result='', is_success=False, is_finished=False)
    ]

    workflow = Plan(tasks=tasks2, goal="Predict abalone age")
    flattened_workflows = flatten_workflow_optimized_v3(workflow)
    for i, sequence in enumerate(flattened_workflows, 1):
        id_list = [task.task_id for task in sequence.tasks]
        print(f"Workflow {i}: {sequence}\n {id_list}")
