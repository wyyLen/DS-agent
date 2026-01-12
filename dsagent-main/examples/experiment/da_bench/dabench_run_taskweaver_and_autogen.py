import asyncio
import sys
from typing import Union

from metagpt.logs import logger

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.const import DA_EVAL_RES_PATH
from util.DABENCH import DABench
from util.common import initialize_record, collect_and_write_result


async def evaluate_all(model: Union["gpt_4o_mini", "glm_4_flash", None], agent=Union["task_weaver", "autogen"], use_reformat=False, batch=""):
    bench = DABench()
    record_path = DA_EVAL_RES_PATH / f"{agent}_{model}{batch}_dabench_onprogress_record.md"
    id_list, predictions, token_cost = initialize_record(record_path)
    # note: predictions of 'task_weaver' and 'autogen' is got from another project, so we don't need to run predict.
    res = bench.eval_all(id_list, predictions, use_reformat=use_reformat)
    res["tokens_cost"] = token_cost
    logger.info(f"res: {res}")
    total_record = [{
        "idx": idx,
        "prediction": prediction
    } for idx, prediction in zip(id_list, predictions)]
    return res, total_record


def main():
    model, agent, use_reformat, batch = "gpt_4o_mini", "task_weaver", False, "_batch1"
    res, ids_predictions = asyncio.run(evaluate_all(model, agent, use_reformat, batch))
    collect_and_write_result(res, ids_predictions, model, agent, use_reformat, batch)


if __name__ == "__main__":
    main()

