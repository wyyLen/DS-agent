import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

from metagpt.strategy.lats_react import LanguageAgentTreeSearch
from examples.experiment.da_bench.util.common import initialize_record
from examples.experiment.ml_benchmark.requirements_prompt import ML_BENCHMARK_REQUIREMENTS
from metagpt.const import DATA_PATH, EXAMPLE_PATH
from metagpt.logs import logger

ML_BENCHMARK_RES_PATH = EXAMPLE_PATH / "experiment" / "ml_benchmark" / "result"


async def run_dsa_lats(agent: LanguageAgentTreeSearch, iterations=5):
    if agent is None:
        raise ValueError("Agent is not initialized.")
    rsp, best_child = await agent.enhance_run(iterations=iterations)
    return rsp

task_list = list(ML_BENCHMARK_REQUIREMENTS.keys())


async def evaluate_ml_benchmark(task_name, data_dir=DATA_PATH):
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/ml_benchmark")):
        raise FileNotFoundError(f"ML-Benchmark dataset not found in {data_dir}.")

    requirement = ML_BENCHMARK_REQUIREMENTS[task_name].format(data_dir=data_dir)
    lats = LanguageAgentTreeSearch(goal=requirement, use_exp_driven_search=True, use_dual_reflection=False)
    await run_dsa_lats(lats, iterations=5)
    prompt_token, completion_token = lats.calculate_total_cost()
    print(f"total token cost: {prompt_token + completion_token}")


def main():
    ml_task_list, cur_idx = list(ML_BENCHMARK_REQUIREMENTS.keys()), 3
    logger.info(f"current ml tasks: {ml_task_list[cur_idx - 1]}")
    asyncio.run(evaluate_ml_benchmark(task_name=ml_task_list[cur_idx - 1]))


if __name__ == "__main__":
    main()
