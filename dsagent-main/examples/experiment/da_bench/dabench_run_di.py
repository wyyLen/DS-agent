import asyncio
import json
import sys
import time
from typing import Union

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from util.DABENCH import DABench
from util.common import initialize_record, collect_and_write_result

from metagpt.const import DA_EVAL_RES_PATH
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.logs import logger
from metagpt.tools.tool_recommend import TypeMatchToolRecommender


async def evaluate_all(model: Union["gpt_4o", "gpt_4o_mini", "glm_4_flash", None], agent="di", use_reformat=True, batch=""):
    bench = DABench()
    record_path = DA_EVAL_RES_PATH / f"{agent}_{model}{batch}_dabench_onprogress_record.md"
    id_list, predictions, token_cost = initialize_record(record_path)
    for key, value in bench.answers.items():
        if key in id_list:
            continue
        requirement = bench.generate_formatted_prompt(key)
        di = DataInterpreter(use_reflection=True, tool_recommender=TypeMatchToolRecommender(tools=["<all>"]))
        try:
            result = await di.run(requirement)
            # 与 MetaGPT 官方设置保持一致
            prediction_json = json.loads(str(result).split("Current Plan")[1].split("## Current Task")[0])
            prediction = prediction_json[-1]["result"]
            predictions.append(prediction)
        except Exception:
            prediction = "some error occurs in DI without any feedback"
            predictions.append(prediction)
            time.sleep(120)
        id_list.append(key)
        token_cost += di.llm.get_costs().total_prompt_tokens + di.llm.get_costs().total_completion_tokens
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
    model, agent, use_reformat, batch = "gpt_4o", "di", True, "_batch1"
    res, ids_predictions = asyncio.run(evaluate_all(model, agent, use_reformat, batch))
    collect_and_write_result(res, ids_predictions, model, agent, use_reformat, batch)


if __name__ == "__main__":
    main()