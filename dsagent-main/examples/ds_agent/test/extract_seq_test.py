from collections import defaultdict, deque


# 示例用法
plan_output = [
    {"task_id": "1", "dependent_task_ids": [], "instruction": "Load and inspect the abalone dataset", "task_type": "pda"},
    {"task_id": "2", "dependent_task_ids": ["1"], "instruction": "Calculate the Pearson correlation coefficient between length and the weight of the whole abalone", "task_type": "correlation analysis"},
    {"task_id": "3", "dependent_task_ids": ["1"], "instruction": "Create a new feature 'volume' by multiplying length, diameter, and height", "task_type": "feature engineering"},
    {"task_id": "4", "dependent_task_ids": ["3"], "instruction": "Split the data into a 70% train set and a 30% test set", "task_type": "data preprocessing"},
    {"task_id": "5", "dependent_task_ids": ["4"], "instruction": "Train a linear regression model to predict the number of rings using original features", "task_type": "model train"},
    {"task_id": "6", "dependent_task_ids": ["5"], "instruction": "Evaluate the model using RMSE on the test set", "task_type": "model evaluate"},
    {"task_id": "7", "dependent_task_ids": ["4"], "instruction": "Train a linear regression model to predict the number of rings using the new 'volume' feature along with original features", "task_type": "model train"},
    {"task_id": "8", "dependent_task_ids": ["7"], "instruction": "Evaluate the model using RMSE on the test set and compare with the previous model", "task_type": "model evaluate"}
]


def extract_task_sequences(plan_output):
    # 构建图和入度表
    graph = defaultdict(list)
    indegree = defaultdict(int)
    tasks = {}

    for task in plan_output:
        task_id = task['task_id']
        dependent_task_ids = task['dependent_task_ids']
        tasks[task_id] = task
        indegree[task_id] = len(dependent_task_ids)
        for dep_id in dependent_task_ids:
            graph[dep_id].append(task_id)

    # 使用拓扑排序提取完整的顺序列表
    def topo_sort():
        queue = deque()
        for task_id, count in indegree.items():
            if count == 0:
                queue.append([task_id])

        all_sequences = []

        while queue:
            sequence = queue.popleft()
            last_task = sequence[-1]
            for next_task in graph[last_task]:
                indegree[next_task] -= 1
                if indegree[next_task] == 0:
                    new_sequence = sequence + [next_task]
                    queue.append(new_sequence)
                    if len(graph[next_task]) == 0:  # 如果没有后续任务了
                        all_sequences.append(new_sequence)

        return all_sequences

    return topo_sort()


if __name__ == '__main__':
    sequences = extract_task_sequences(plan_output)
    print(sequences)
