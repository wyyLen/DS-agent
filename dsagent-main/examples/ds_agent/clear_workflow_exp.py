"""
由于从kaggle解决方案中提取的workflow存在大量与核心逻辑无关的代码，需要进行清洗
case:
    1. 结果提交和文件处理 [
        {
            "instruction": "Prepare the final predictions from the best-performing model for submission by creating a DataFrame with an 'Id' column and a 'Sales' column, and export this DataFrame to a CSV file.",
            "task_type": "other"
        },
        {
            "instruction": "Prepare the final submission by averaging the predictions from the ensemble.",
            "task_type": "other"
        },
        {
            "instruction": "Submit the predictions to the competition and review the model's performance on the public leaderboard.",
            "task_type": "other"
        },
        {
            "instruction": "Generate the final submission file with the ensemble predictions and prepare it for submission according to the competition's format requirements.",
            "task_type": "other"
        }
    ]

    2. 环境配置 [
        {
            "instruction": "Initialize the competition environment using 'enefit.make_env()' to handle data streaming and predictions.",
            "task_type": "other"
        }
    ]

"""
import json
import re

from metagpt.const import EXAMPLE_DATA_PATH


def clear_workflow_exp_from_kaggle(workflow: list):
    cleaned_workflow = []
    task_id_mapping = {}

    def is_task_to_remove(task: dict):
        keyword_list = ["prepare the final", "generate the final", "final submission", "submit the predictions"]
        pattern = r'(initialize.*environment|set.*up.*environment)'
        if "other" in task["task_type"]:
            if any(keyword in str(task["instruction"]).lower() for keyword in keyword_list):
                return True
            if re.search(pattern, task["instruction"].lower()):
                return True
        return False

    for task in workflow:
        task_id = task["task_id"]
        if not is_task_to_remove(task):
            cleaned_workflow.append(task)
            task_id_mapping[task_id] = len(cleaned_workflow)

    for new_id, task in enumerate(cleaned_workflow, start=1):
        task["task_id"] = str(new_id)
        original_dependent_task_ids = task["dependent_task_ids"]

        def get_dependency_task_ids(original_dependent_task_ids, task_id_mapping):
            dependency_task_ids = []
            for original_id in original_dependent_task_ids:
                if original_id in task_id_mapping:
                    dependency_task_ids.append(str(task_id_mapping[original_id]))
            return dependency_task_ids

        task["dependent_task_ids"] = get_dependency_task_ids(original_dependent_task_ids, task_id_mapping)

    return cleaned_workflow


def main():
    workflow_path = EXAMPLE_DATA_PATH / "exp_bank" / "workflow_exp2.json"
    workflow_clean_path = EXAMPLE_DATA_PATH / "exp_bank" / "workflow_exp2_clean.json"
    with open(workflow_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        workflow = item["workflow"]
        cleaned_workflow = clear_workflow_exp_from_kaggle(workflow)
        item["workflow"] = cleaned_workflow
    with open(workflow_clean_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
