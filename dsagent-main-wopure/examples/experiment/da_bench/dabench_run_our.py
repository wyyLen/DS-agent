import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Union

from metagpt.logs import logger
from metagpt.roles.ds_agent.ds_agent import DSAgent

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.const import DA_EVAL_RES_PATH, EXAMPLE_PATH
from util.DABENCH import DABench
from util.common import check_file_exist, initialize_record, collect_and_write_result


async def run_ds_agent(agent: DSAgent, requirement: str):
    if agent is None:
        raise ValueError("Agent is not initialized.")
    rsp = await agent.run(requirement)
    return rsp.content


async def evaluate_all(model: Union["gpt_4o", "gpt_4o_mini", "glm_4_flash", None], agent="our_agent", use_reformat=False, batch="", mode=""):
    bench = DABench()
    ds_agent = DSAgent(use_reflection=True, use_rag=True, use_kaggle_exp=True, use_exp_extractor=False)
    record_path = DA_EVAL_RES_PATH / f"{agent}_{model}{mode}{batch}_dabench_onprogress_record.md"
    id_list, predictions, token_cost = initialize_record(record_path)
    if model == "gpt_4o_mini":
        ds_agent.llm.cost_manager.update_cost(token_cost, 0, model.replace("_", "-") + "-2024-07-18")
    elif model == "gpt_4o":
        ds_agent.llm.cost_manager.update_cost(token_cost, 0, model.replace("_", "-") + "-2024-11-20")
    else:
        ds_agent.llm.cost_manager.update_cost(token_cost, 0, model.replace("_", "-"))
    for key, value in bench.answers.items():
        if key in id_list:
            continue
        requirement = bench.generate_formatted_prompt(key)
        try:
            rsp = await run_ds_agent(ds_agent, requirement)
        except Exception as e:
            rsp = f"some error occurs in our-agent without any feedback {e}"
            print(rsp)
            raise e
        id_list.append(key)
        predictions.append(rsp)
        # predictions.append(json.loads(str(rsp).split("Current Plan")[1].split("## Current Task")[0])[-1]["result"])
        token_cost = ds_agent.llm.get_costs().total_prompt_tokens + ds_agent.llm.get_costs().total_completion_tokens
        with open(record_path, "w") as f:
            f.write(json.dumps(id_list) + "\n")
            f.write(json.dumps(predictions) + "\n")
            f.write(str(token_cost) + "\n")
        ds_agent.clear_content()
    res = bench.eval_all(id_list, predictions, use_reformat=use_reformat)
    res["tokens_cost"] = token_cost
    logger.info(f"res: {res}")
    total_record = [{
        "idx": idx,
        "prediction": prediction
    } for idx, prediction in zip(id_list, predictions)]
    return res, total_record


def main():
    # model, agent, use_reformat, batch, mode = "glm_4_flash", "DSAInterpreter", False, "_batch3", ""
    model, agent, use_reformat, batch, mode = "glm_4_flash", "our_agent", True, "_batch0", "_lats"
    res, ids_predictions = asyncio.run(evaluate_all(model, agent, use_reformat, batch, mode))
    collect_and_write_result(res, ids_predictions, model, agent, use_reformat, batch, mode)


if __name__ == "__main__":
    main()
