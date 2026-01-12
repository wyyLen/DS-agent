import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Union

from metagpt.logs import logger
from metagpt.strategy.lats_react import LanguageAgentTreeSearch

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.const import DA_EVAL_RES_PATH, EXAMPLE_PATH
from util.DABENCH import DABench
from util.common import check_file_exist, initialize_record, collect_and_write_result


async def run_dsa_lats(agent: LanguageAgentTreeSearch, iterations=5):
    if agent is None:
        raise ValueError("Agent is not initialized.")
    rsp, best_child = await agent.enhance_run(iterations=iterations)
    return rsp


async def evaluate_all(model: Union["gpt_4o", "gpt_4o_mini", "glm_4_flash", None], agent="our_agent", use_reformat=False, batch="", mode=""):
    bench = DABench()
    record_path = DA_EVAL_RES_PATH / f"{agent}_{model}{mode}{batch}_dabench_onprogress_record.md"
    id_list, predictions, token_cost = initialize_record(record_path)
    for key, value in bench.answers.items():
        if key in id_list:
            continue
        requirement = bench.generate_formatted_prompt(key)
        lats = LanguageAgentTreeSearch(goal=requirement)
        try:
            rsp = await run_dsa_lats(lats, 5)
        except Exception as e:
            rsp = f"some error occurs in LATS without any feedback {e}"
            print(rsp)
            raise e
        id_list.append(key)
        predictions.append(rsp)
        prompt_token, completion_token = lats.calculate_total_cost()
        print(f"lats result: {rsp}, token_cost: {prompt_token + completion_token}")
        token_cost += prompt_token + completion_token
        with open(record_path, "w") as f:
            f.write(json.dumps(id_list) + "\n")
            f.write(json.dumps(predictions) + "\n")
            f.write(str(token_cost) + "\n")
    res = bench.eval_all(id_list, predictions, use_reformat=use_reformat)
    res["tokens_cost"] = token_cost
    logger.info(f"res: {res}")
    total_record = [{
        "idx": idx,
        "prediction": prediction
    } for idx, prediction in zip(id_list, predictions)]
    return res, total_record


def main():
    model, agent, use_reformat, batch, mode = "gpt_4o_mini", "lats", True, "_batch1", ""
    res, ids_predictions = asyncio.run(evaluate_all(model, agent, use_reformat, batch, mode))
    collect_and_write_result(res, ids_predictions, model, agent, use_reformat, batch, mode)


if __name__ == "__main__":
    main()
