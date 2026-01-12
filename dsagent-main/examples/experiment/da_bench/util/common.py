import json
from pathlib import Path

from dsagent_core.const import DA_EVAL_RES_PATH
from metagpt.logs import logger


def check_file_exist(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)


def initialize_record(record_path):
    check_file_exist(Path(record_path))
    with open(record_path, "r") as f:
        record_data = f.read().splitlines()
    if record_data:
        id_list = json.loads(record_data[0])
        predictions = json.loads(record_data[1])
        token_cost = int(record_data[2])
    else:
        id_list, predictions, token_cost = [], [], 0
    return id_list, predictions, token_cost


def collect_and_write_result(res, ids_predictions, model, agent, use_reformat, batch, mode=""):
    output_data = {
        "res": res,
        "ids_predictions": ids_predictions
    }
    logger.info(output_data)
    if use_reformat:
        result_path = DA_EVAL_RES_PATH / f"{agent}_{model}{mode}{batch}_dabench_result_reformat.json"
    else:
        result_path = DA_EVAL_RES_PATH / f"{agent}_{model}{mode}{batch}_dabench_result.json"
    check_file_exist(Path(result_path))
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)


def record_token_cost(file_path, prompt_token_cost, completion_token_cost):
    check_file_exist(Path(file_path))
    with open(file_path, "r") as f:
        record_data = f.read().splitlines()

    if record_data:
        lats_count = int(record_data[0])
        prompt_cost = int(record_data[1])
        completion_cost = int(record_data[2])
    else:
        lats_count = 0
        prompt_cost = 0
        completion_cost = 0
    lats_count += 1
    prompt_cost += prompt_token_cost
    completion_cost += completion_token_cost
    with open(file_path, "w") as f:
        f.write(str(lats_count) + "\n")
        f.write(str(prompt_cost) + "\n")
        f.write(str(completion_cost) + "\n")


