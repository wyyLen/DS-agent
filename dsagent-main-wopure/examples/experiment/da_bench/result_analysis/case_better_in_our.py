import json
import os
from typing import List, Dict, Set
from pathlib import Path

from metagpt.const import DA_EVAL_RES_PATH, question_file


def analyze_test_results(our_method_files: List[str], comparison_method_files: List[str]) -> List[int]:
    """
    分析测试结果，找出我们的方法正确而对比方法错误的案例。

    参数:
        our_method_files: 我们方法的测试结果JSON文件路径列表
        comparison_method_files: 对比方法的测试结果JSON文件路径列表

    返回:
        我们的方法正确而对比方法错误的案例ID列表
    """
    # 存储我们方法和对比方法正确的案例ID
    our_correct_ids = set()
    comparison_correct_ids = set()

    # 分析我们的方法结果
    for file_path in our_method_files:
        correct_ids = get_correct_test_cases(file_path)
        our_correct_ids.update(correct_ids)

    # 分析对比方法结果
    for file_path in comparison_method_files:
        correct_ids = get_correct_test_cases(file_path)
        comparison_correct_ids.update(correct_ids)

    # 找出我们方法正确而对比方法错误的案例
    better_case_ids = list(our_correct_ids - comparison_correct_ids)
    better_case_ids.sort()  # 按ID排序

    return better_case_ids


def get_correct_test_cases(file_path: str) -> Set[int]:
    # 从单个JSON文件中获取所有正确的测试案例ID
    correct_ids = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'res' in data and 'results' in data['res']:
            for result in data['res']['results']:
                if is_test_case_correct(result):
                    correct_ids.add(result['id'])
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

    return correct_ids


def is_test_case_correct(test_case: Dict) -> bool:
    # 判断单个测试案例是否正确
    if 'correctness' not in test_case:
        return False
    # 如果correctness中有任何一项不为True，则认为测试不正确
    return all(test_case['correctness'].values())


def load_case_difficulties(question_path: str) -> Dict[int, str]:
    """加载案例难度信息"""
    case_difficulty = {}
    with open(question_path, 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line)
            case_difficulty[case['id']] = case.get('level', 'unknown')  # 避免KeyError
    return case_difficulty


def get_better_cases_in_our_method(our_method_filenames: List[str], comparison_method_filenames: List[str],
                                  case_difficulty_map: Dict[int, str]):
    our_method_files = [str(DA_EVAL_RES_PATH / filename) for filename in our_method_filenames]
    comparison_method_files = [str(DA_EVAL_RES_PATH / filename) for filename in comparison_method_filenames]
    better_case_ids = analyze_test_results(our_method_files, comparison_method_files)
    print(f"我们的方法比对比方法多正确了 {len(better_case_ids)} 个案例")
    print(f"这些案例的ID是: {better_case_ids}")

    # 分析难度分布
    difficulty_info = {}  # 使用字典存储数量和ID列表
    for case_id in better_case_ids:
        level = case_difficulty_map.get(case_id, 'unknown')  # 处理未知难度
        if level not in difficulty_info:
            difficulty_info[level] = {'count': 0, 'ids': []}
        difficulty_info[level]['count'] += 1
        difficulty_info[level]['ids'].append(case_id)

    total = len(better_case_ids)
    if total > 0:
        print("难度分布统计:")
        for level, info in difficulty_info.items():
            count = info['count']
            ids = info['ids']
            percentage = (count / total) * 100
            print(f"- {level}: {count} 个案例 ({percentage:.1f}%)   对应ID列表：{ids}")
    else:
        print("没有案例需要统计难度分布。")

    return better_case_ids


def main():
    case_difficulty_map = load_case_difficulties(question_file)

    our_w_plan_method_filenames = [
        "our_agent_gpt_4o_mini_batch1_dabench_result_reformat.json",
        "our_agent_gpt_4o_mini_dabench_result_reformat.json"
    ]
    our_w_lats_filenames = [
        "lats_gpt_4o_mini_batch0_dabench_result_reformat.json",
        "lats_gpt_4o_mini_batch1_dabench_result_reformat.json"
    ]
    comparison_method_filenames = [
        "infiagent_deepseek_r1_batch1_redo2_dabench_result_reformat-check.json",
        "infiagent_o1_batch0_dabench_result_reformat4k-checked.json"
    ]

    print("W-Plan method:")
    get_better_cases_in_our_method(our_w_plan_method_filenames, comparison_method_filenames, case_difficulty_map)
    print("\nW-Lats method:")
    get_better_cases_in_our_method(our_w_lats_filenames, comparison_method_filenames, case_difficulty_map)


if __name__ == "__main__":
    main()