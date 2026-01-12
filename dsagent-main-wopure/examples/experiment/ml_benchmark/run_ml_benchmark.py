import asyncio
import json
import os

import fire

from examples.experiment.da_bench.util.common import initialize_record
from examples.experiment.ml_benchmark.requirements_prompt import ML_BENCHMARK_REQUIREMENTS
from metagpt.const import DATA_PATH, EXAMPLE_PATH
from metagpt.logs import logger
from metagpt.roles.ds_agent.ds_agent import DSAgent


ML_BENCHMARK_RES_PATH = EXAMPLE_PATH / "experiment" / "ml_benchmark" / "result"


async def run_ds_agent(agent: DSAgent, requirement: str):
    if agent is None:
        raise ValueError("Agent is not initialized.")
    rsp = await agent.run(requirement)
    return rsp.content


task_list = list(ML_BENCHMARK_REQUIREMENTS.keys())


async def batch_evaluate_ml_benchmark(model, agent, batch, mode, data_dir=DATA_PATH):
    # note: @deprecated
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/ml_benchmark")):
        raise FileNotFoundError(f"ML-Benchmark dataset not found in {data_dir}.")

    record_path = ML_BENCHMARK_RES_PATH / f"{agent}_{model}{mode}{batch}_ml_bench_record.md"
    id_list, predictions, token_cost = initialize_record(record_path)

    ds_agent = DSAgent(use_reflection=True, use_rag=True, use_kaggle_exp=True, use_exp_extractor=False)
    if model == "gpt_4o_mini":
        ds_agent.llm.cost_manager.update_cost(token_cost, 0, model.replace("_", "-") + "-2024-07-18")
    else:
        ds_agent.llm.cost_manager.update_cost(token_cost, 0, model.replace("_", "-"))

    for task_name in list(ML_BENCHMARK_REQUIREMENTS.keys()):
        if task_name in id_list:
            continue
        requirement = ML_BENCHMARK_REQUIREMENTS[task_name].format(data_dir=data_dir)
        try:
            rsp = await run_ds_agent(ds_agent, requirement)
        except Exception:
            rsp = "some error occurs in DSA-Interpreter without any feedback"
        id_list.append(task_name)
        predictions.append(rsp)
        token_cost = ds_agent.llm.get_costs().total_prompt_tokens + ds_agent.llm.get_costs().total_completion_tokens
        with open(record_path, "w") as f:
            f.write(json.dumps(id_list) + "\n")
            f.write(json.dumps(predictions) + "\n")
            f.write(str(token_cost) + "\n")
        ds_agent.clear_content()


async def evaluate_ml_benchmark(task_name, data_dir=DATA_PATH):
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/ml_benchmark")):
        raise FileNotFoundError(f"ML-Benchmark dataset not found in {data_dir}.")

    ds_agent = DSAgent(use_reflection=True, use_rag=True, use_kaggle_exp=True, use_exp_extractor=False)
    requirement = ML_BENCHMARK_REQUIREMENTS[task_name].format(data_dir=data_dir)
    await run_ds_agent(ds_agent, requirement)
    token_cost = ds_agent.llm.get_costs().total_prompt_tokens + ds_agent.llm.get_costs().total_completion_tokens
    print(f"total token cost: {token_cost}")


def main():
    ml_task_list, cur_idx = list(ML_BENCHMARK_REQUIREMENTS.keys()), 2
    logger.info(f"current ml tasks: {ml_task_list[cur_idx - 1]}")
    asyncio.run(evaluate_ml_benchmark(task_name=ml_task_list[cur_idx - 1]))


if __name__ == "__main__":
    fire.Fire(main)
