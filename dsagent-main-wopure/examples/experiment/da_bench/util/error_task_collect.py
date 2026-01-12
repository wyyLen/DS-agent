import json
from collections import Counter
from typing import Union

from metagpt.const import DA_EVAL_RES_PATH


def count_and_sort_numbers(*lists):
    # 将所有列表合并为一个大列表
    merged_list = [num for lst in lists for num in lst]
    # 使用 Counter 统计每个数字出现的次数
    count = Counter(merged_list)
    # 按出现次数从高到低排序
    sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return sorted_count


def get_wrong_task_ids(model: Union["gpt_4o_mini", "glm_4_flash", None], agent="our_agent", batch="", mode=""):
    result_path = DA_EVAL_RES_PATH / f"{agent}_{model}{mode}{batch}_dabench_result_reformat.json"
    with open(result_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    result_list = data["res"]["results"]
    wrong_task_ids = []
    for result in result_list:
        if not ("correctness" in result and all(result["correctness"].values())):
            wrong_task_ids.append(result["id"])
    print(f"{agent}_{model}{mode}{batch}_reformat: \n"
          f"wrong_task_ids: {wrong_task_ids}")
    return wrong_task_ids


if __name__ == '__main__':
    # note: task_weaver
    print("------------------------------------------- result of task_weaver: ")
    model, agent, batch, mode = "gpt_4o_mini", "task_weaver", "", ""
    task_weaver_list1 = get_wrong_task_ids(model, agent, batch, mode)
    task_weaver_list2 = get_wrong_task_ids("glm_4_flash", agent, batch, mode)

    # note: DI
    print("------------------------------------------- result of DI: ")
    model, agent, batch, mode = "gpt_4o_mini", "DI", "_batch1", ""
    DI_list1 = get_wrong_task_ids(model, agent, batch, mode)
    DI_list2 = get_wrong_task_ids(model, agent, "", mode)
    DI_list3 = get_wrong_task_ids("glm_4_flash", agent, "_batch1", mode)
    DI_list4 = get_wrong_task_ids("glm_4_flash", agent, "_batch2", mode)

    # note: our_agent
    print("------------------------------------------- result of our_agent: ")
    model, agent, batch, mode = "gpt_4o_mini", "our_agent", "_batch1", ""
    our_agent_list1 = get_wrong_task_ids(model, agent, batch, mode)
    our_agent_list2 = get_wrong_task_ids(model, agent, "", mode)
    our_agent_list3 = get_wrong_task_ids("glm_4_flash", agent, batch, mode)
    our_agent_list4 = get_wrong_task_ids("glm_4_flash", agent, "_batch2", mode)

    print("------------------------------------------- result of _extractor: ")
    _extractor_list1 = get_wrong_task_ids("gpt_4o_mini", "our_agent", "_batch0", "_extractor")
    _extractor_list2 = get_wrong_task_ids("gpt_4o_mini", "our_agent", "_batch0", "_extractor_after")
    _extractor_list3 = get_wrong_task_ids("glm_4_flash", "our_agent", "_batch0", "_extractor_after")

    result = count_and_sort_numbers(task_weaver_list1, task_weaver_list2, DI_list1, DI_list2, DI_list3, DI_list4, our_agent_list1, our_agent_list2, our_agent_list3, our_agent_list4, _extractor_list1, _extractor_list2, _extractor_list3)
    print(result)
