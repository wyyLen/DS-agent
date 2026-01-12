import json
from typing import Union

from DABENCH import DABench, evaluate_accuracy_by_question, evaluate_accuracy_by_sub_question, \
    evaluate_accuracy_proportional_by_sub_question_adjusted, evaluate_completeness_by_question
from common import initialize_record
from metagpt.const import DA_EVAL_RES_PATH
from metagpt.logs import logger


def get_hard_tasks():
    bench = DABench()
    hard_task_ids = []
    for key, value in bench.questions.items():
        level = bench.get_task_level(key)
        if level == 'hard':
            hard_task_ids.append(key)
    return hard_task_ids


def get_hard_tasks_res_from_original_md(model: Union["gpt_4o_mini", "glm_4_flash", None], agent="our_agent", use_reformat=False, batch=1, mode=""):
    bench = DABench()
    hard_task_ids, hard_task_predictions = get_hard_tasks(), []
    record_path = DA_EVAL_RES_PATH / f"{agent}_{model}{mode}_batch{batch}_dabench_onprogress_record.md"
    id_list, predictions, token_cost = initialize_record(record_path)
    for task_id, prediction in zip(id_list, predictions):
        if task_id in hard_task_ids:
            hard_task_predictions.append(prediction)
    res = bench.eval_all(hard_task_ids, hard_task_predictions)
    res["tokens_cost"] = token_cost
    logger.info(f"res: {res}")


def get_hard_tasks_res(model: Union["gpt_4o_mini", "glm_4_flash", None], agent="our_agent", batch="", mode="", reformat=True):
    hard_task_ids = get_hard_tasks()
    result_path = DA_EVAL_RES_PATH / f"{agent}_{model}{mode}{batch}_dabench_result_reformat.json" if reformat else DA_EVAL_RES_PATH / f"{agent}_{model}{mode}{batch}_dabench_result.json"
    with open(result_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    result_list = data["res"]["results"] if "results" in data["res"] else []
    results = []
    for result in result_list:
        if result["id"] in hard_task_ids:
            results.append(result)
    accuracy_by_question = evaluate_accuracy_by_question(results)
    accuracy_by_sub_question = evaluate_accuracy_by_sub_question(results)
    proportional_accuracy_by_sub_question = evaluate_accuracy_proportional_by_sub_question_adjusted(results)
    print(f"{agent}_{model}{mode}{batch}_reformat: \n"
          f"accuracy_by_question: {accuracy_by_question} \n"
          f"accuracy_by_sub_question: {accuracy_by_sub_question} \n"
          f"proportional_accuracy_by_sub_question: {proportional_accuracy_by_sub_question} \n")


if __name__ == '__main__':
    print(f"hard tasks: {get_hard_tasks()}")
    use_reformat = True

    # note: autogen
    print("------------------------------------------- result of autogen: ")
    get_hard_tasks_res("gpt_4o_mini", "autogen", "_batch0", "", use_reformat)
    get_hard_tasks_res("gpt_4o_mini", "autogen", "_batch1", "", use_reformat)
    get_hard_tasks_res("glm_4_flash", "autogen", "_batch0", "", use_reformat)
    get_hard_tasks_res("glm_4_flash", "autogen", "_batch1", "", use_reformat)

    # note: task_weaver
    print("------------------------------------------- result of task_weaver: ")
    get_hard_tasks_res("gpt_4o_mini", "task_weaver", "", "", use_reformat)
    get_hard_tasks_res("gpt_4o_mini", "task_weaver", "_batch1", "", use_reformat)
    get_hard_tasks_res("glm_4_flash", "task_weaver", "", "", use_reformat)
    get_hard_tasks_res("glm_4_flash", "task_weaver", "_batch1", "", use_reformat)

    # note: DI
    print("------------------------------------------- result of DI: ")
    get_hard_tasks_res("gpt_4o_mini", "DI", "_batch1", "", use_reformat)
    get_hard_tasks_res("gpt_4o_mini", "DI", "", "", use_reformat)
    get_hard_tasks_res("glm_4_flash", "DI", "_batch1", "", use_reformat)
    get_hard_tasks_res("glm_4_flash", "DI", "_batch2", "", use_reformat)

    # note: our_agent
    print("------------------------------------------- result of our_agent: ")
    get_hard_tasks_res("gpt_4o_mini", "our_agent", "_batch1", "", use_reformat)
    get_hard_tasks_res("gpt_4o_mini", "our_agent", "", "", use_reformat)
    get_hard_tasks_res("glm_4_flash", "our_agent", "_batch1", "", use_reformat)
    get_hard_tasks_res("glm_4_flash", "our_agent", "_batch2", "", use_reformat)

    # note: lats
    print("------------------------------------------- result of lats: ")
    get_hard_tasks_res("gpt_4o_mini", "lats", "_batch0", "", use_reformat)
    get_hard_tasks_res("gpt_4o_mini", "lats", "_batch1", "", use_reformat)
    get_hard_tasks_res("glm_4_flash", "lats", "_batch0", "", use_reformat)

    print("------------------------------------------- result of _extractor: ")
    get_hard_tasks_res("gpt_4o_mini", "our_agent", "_batch0", "_extractor", use_reformat)
    get_hard_tasks_res("gpt_4o_mini", "our_agent", "_batch0", "_extractor_after", use_reformat)
    get_hard_tasks_res("glm_4_flash", "our_agent", "_batch0", "_extractor_after", use_reformat)

