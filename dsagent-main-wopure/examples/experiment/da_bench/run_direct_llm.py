import asyncio
import json
import re
import sys
from typing import Union

from examples.build_customized_agent import RunnableCoder
from examples.react_agent import ReactAgent
from examples.react_runner_agent import ReactRunnerAgent
from metagpt.config2 import Config
from metagpt.logs import logger

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.const import DA_EVAL_RES_PATH, METAGPT_ROOT
from util.DABENCH import DABench
from util.common import initialize_record, collect_and_write_result


def get_llm_config(model: Union["gpt_4o","gpt_4o_mini", "glm_4_flash", None]):
    config = None
    if model == "gpt_4o":
        gpt4o_config_path = METAGPT_ROOT / "config" / "gpt-4o.yaml"
        config = Config.from_yaml_file(gpt4o_config_path)
    elif model == "gpt_4o_mini":
        gpt4o_config_path = METAGPT_ROOT / "config" / "gpt-4o-mini.yaml"
        config = Config.from_yaml_file(gpt4o_config_path)
    else:
        raise NotImplementedError
    return config


async def evaluate_all(model: Union["o1", "o3_mini", "deepseek_r1", "gpt_4o", "gpt_4o_mini", "glm_4_flash", None], agent: str, use_reformat: bool, batch=""):
    bench = DABench()
    record_path = DA_EVAL_RES_PATH / f"{agent}_{model}{batch}_dabench_onprogress_record.md"
    id_list, predictions, token_cost = initialize_record(record_path)
    for key, value in bench.answers.items():
        if key in id_list:
            continue
        agent = ReactAgent(config=get_llm_config(model))
        requirement = bench.generate_formatted_prompt(key)
        rsp = await agent.run(requirement)
        id_list.append(key)
        predictions.append(rsp.content)
        logger.info(f"agent.private_llm.cost_manager: {agent.private_llm.cost_manager}")
        logger.info(f"agent.llm.cost_manager: {agent.llm.cost_manager}")
        token_cost += agent.private_llm.get_costs().total_prompt_tokens + agent.private_llm.get_costs().total_completion_tokens
        print(f"rsp: {rsp.content} \ntoken_cost: {token_cost}")
        with open(record_path, "w") as f:
            f.write(json.dumps(id_list) + "\n")
            f.write(json.dumps(predictions) + "\n")
            f.write(str(token_cost) + "\n")
    res = bench.eval_all(id_list, predictions, use_reformat=use_reformat)
    logger.info(f"res: {res}")
    total_record = [{
        "idx": idx,
        "prediction": prediction
    } for idx, prediction in zip(id_list, predictions)]
    return res, total_record


async def evaluate_all_result_from_infiagent(model: Union["o1", "o3_mini", "deepseek_r1", "gpt_4o", "gpt_4o_mini", "glm_4_flash", None], agent: str, use_reformat: bool, batch=""):
    bench = DABench()
    record_path = DA_EVAL_RES_PATH / f"{agent}_{model}{batch}.json"
    print(f"record_path: {record_path}")
    with open(record_path, "r") as f:
        data = json.load(f)
    id_list, predictions = [], []
    for item in data:
        id_list.append(item["id"])
        if model == "deepseek_r1":
            cleaned_response = remove_think_tags(item["response"])
            predictions.append(cleaned_response)
        else:
            predictions.append(truncate_long_string(item["response"]))
    res = bench.eval_all(id_list, predictions, use_reformat=use_reformat)
    logger.info(f"res: {res}")
    total_record = [{
        "idx": idx,
        "prediction": prediction
    } for idx, prediction in zip(id_list, predictions)]
    return res, total_record


def remove_think_tags(response: str) -> str:
    """
    使用正则表达式移除所有 <think>...</think> 块（包括跨行内容）
    并返回清理后的字符串，保留标签外的内容
    """
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def truncate_long_string(input_str: str, max_length: int = 4096) -> str:
    # desc 当输入字符串长度超过 max_length 字符时，保留最后 max_length 个字符
    return input_str[-max_length:] if len(input_str) > max_length else input_str


def main():
    model, agent, use_reformat, batch = "deepseek_r1", "infiagent", True, "_batch1_redo2"
    res, ids_predictions = asyncio.run(evaluate_all_result_from_infiagent(model, agent, use_reformat, batch))
    collect_and_write_result(res, ids_predictions, model, agent, use_reformat, batch)


if __name__ == '__main__':
    main()
