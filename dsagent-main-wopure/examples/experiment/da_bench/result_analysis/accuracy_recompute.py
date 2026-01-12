import json

from examples.experiment.da_bench.util.DABENCH import evaluate_accuracy_by_question, evaluate_accuracy_by_sub_question, \
    evaluate_accuracy_proportional_by_sub_question_adjusted
from metagpt.const import DA_EVAL_RES_PATH


def recompute_accuracy(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    res = json.loads(data)
    final_res = res["res"]["results"]
    accuracy_by_question = evaluate_accuracy_by_question(final_res)
    accuracy_by_sub_question = evaluate_accuracy_by_sub_question(final_res)
    proportional_accuracy_by_sub_question = evaluate_accuracy_proportional_by_sub_question_adjusted(final_res)

    return {
        "accuracy_by_question": accuracy_by_question,
        "accuracy_by_sub_question": accuracy_by_sub_question,
        "proportional_accuracy_by_sub_question": proportional_accuracy_by_sub_question,
    }


if __name__ == '__main__':
    # di
    di_4o_path = DA_EVAL_RES_PATH / "di_gpt_4o_batch0_dabench_result_reformat.json"
    di_4o_path1 = DA_EVAL_RES_PATH / "di_gpt_4o_batch1_dabench_result_reformat.json"
    di_4o_mini_path = DA_EVAL_RES_PATH / "di_gpt_4o_mini_dabench_result_reformat.json"
    di_4o_mini_path1 = DA_EVAL_RES_PATH / "di_gpt_4o_mini_batch1_dabench_result_reformat.json"
    di_glm4_flash_path1 = DA_EVAL_RES_PATH / "di_glm_4_flash_batch1_dabench_result_reformat.json"
    di_glm4_flash_path2 = DA_EVAL_RES_PATH / "di_glm_4_flash_batch2_dabench_result_reformat.json"
    # infiagent
    infi_r1_final_path = DA_EVAL_RES_PATH / "infiagent_deepseek_r1_batch1_redo2_dabench_result_reformat-check.json"
    infi_o1_final_path = DA_EVAL_RES_PATH / "infiagent_o1_batch0_dabench_result_reformat4k-checked.json"
    infi_o3_mini_final_path = DA_EVAL_RES_PATH / "infiagent_o3_mini_batch1_dabench_result_reformat4k.json"
    # our w-plan
    our_4o_final_path = DA_EVAL_RES_PATH / "our_agent_gpt_4o_batch1_dabench_result_reformat.json"
    our_4o_mini_lats_path = DA_EVAL_RES_PATH / "our_agent_gpt_4o_mini_lats_batch0_dabench_result_reformat.json"
    our_glm4_flash_lats_path = DA_EVAL_RES_PATH / "our_agent_glm_4_flash_lats_batch0_dabench_result_reformat.json"
    our_glm4_flash_path1 = DA_EVAL_RES_PATH / "our_agent_glm_4_flash_batch1_dabench_result_reformat.json"
    our_glm4_flash_path2 = DA_EVAL_RES_PATH / "our_agent_glm_4_flash_batch2_dabench_result_reformat.json"
    # our w-lats
    lats_4o_mini_path = DA_EVAL_RES_PATH / "lats_gpt_4o_mini_batch0_dabench_result_reformat.json"
    lats_4o_mini_path2 = DA_EVAL_RES_PATH / "lats_gpt_4o_mini_batch1_dabench_result_reformat.json"
    lats_glm4_flash_path = DA_EVAL_RES_PATH / "lats_glm_4_flash_batch0_dabench_result_reformat.json"
    lats_4o_mini_ablation_path = DA_EVAL_RES_PATH / "lats_gpt_4o_mini_ablation_batch0_dabench_result_reformat.json"
    acc = recompute_accuracy(di_4o_path)
    print(acc)
