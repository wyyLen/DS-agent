import os

import fire

from examples.di.requirements_prompt import ML_BENCHMARK_REQUIREMENTS
from metagpt.const import DATA_PATH
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.tools.tool_recommend import TypeMatchToolRecommender

ml_task_list, cur_idx = list(ML_BENCHMARK_REQUIREMENTS.keys()), 7


# Ensure ML-Benchmark dataset has been downloaded before using these example.
async def main(task_name=ml_task_list[cur_idx - 1], data_dir=DATA_PATH, use_reflection=True):
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/ml_benchmark")):
        raise FileNotFoundError(f"ML-Benchmark dataset not found in {data_dir}.")

    requirement = ML_BENCHMARK_REQUIREMENTS[task_name].format(data_dir=data_dir)
    di = DataInterpreter(use_reflection=use_reflection, tool_recommender=TypeMatchToolRecommender(tools=["<all>"]))
    try:
        await di.run(requirement)
    except Exception:
        print("some error occurs in DSA-Interpreter without any feedback")
    token_cost = di.llm.get_costs().total_prompt_tokens + di.llm.get_costs().total_completion_tokens
    print(f"total token cost: {token_cost}")


if __name__ == "__main__":
    fire.Fire(main)