import asyncio
import json
import os
import re
import subprocess
import sys

from examples.ds_agent.kaggle_desc import (
    titanic_desc, python_and_analyze_data_final_project, tabular_media_campaign_cost, tabular_wild_blueberry_yield,
    commonlit_evaluate_student_summaries, feedback_prize_2021
)
from metagpt.actions.ds_agent.extract_thought import ThoughtExtract
from metagpt.actions.ds_agent.fixed_plan_for_test import FixedPlan, get_fixed_plan
from metagpt.actions.ds_agent.query_utils import QueryUtils
from metagpt.config2 import Config
from metagpt.logs import logger
from metagpt.roles.ds_agent.ds_agent import add_to_exp_bank_with_metadata, add_to_workflow_exp_bank

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.const import EXAMPLE_DATA_PATH, METAGPT_ROOT, EXP_PLAN, WORKFLOW_EXP

gpt4turbo_config_path = METAGPT_ROOT / "config" / "gpt-4-turbo.yaml"
gpt4t_config = Config.from_yaml_file(gpt4turbo_config_path)

kaggle_nlp = [
    {
        "slug": "commonlit-evaluate-student-summaries",   # desc: 构建一个模型，评估学生文本摘要的分数（表达程度、清晰度、流畅度）
        "goal": commonlit_evaluate_student_summaries,
    },
    {
        "slug": "feedback-prize-2021",                    # desc: 开发一个模型来对学生的文章中的论证和修辞元素进行细分和分类
        "goal": feedback_prize_2021,
    }
]

kaggle_tabular = [
    {
        "slug": "playground-series-s3e11",      # desc: Regression - 根据媒体营销活动成本数据集，预测测试集中目标的成本
        "goal": tabular_media_campaign_cost,
    },
    {
        "slug": "playground-series-s3e14",      # desc:Regression - 根据野生蓝莓数据集，使用回归来预测野生蓝莓的产量
        "goal": tabular_wild_blueberry_yield,
    },
    {
        'slug': 'titanic',                      # desc: Classification - 预测泰坦尼克号乘客生还率
        'goal': titanic_desc,
    },
    {
        "slug": "python-and-analyze-data-final-project",     # desc: Classification - 是否可以使用银行卡上的收支信息来预测客户的性别？如果是这样，这种预测的准确性如何？
        "goal": python_and_analyze_data_final_project,
    }
]

kaggle_cases = [
    {
        "type": "nlp",
        "list": kaggle_nlp,
    },
    {
        "type": "tabular",
        "list": kaggle_tabular,
    },
]

KAGGLE_CASE_PATH = EXAMPLE_DATA_PATH / "kaggle_code"


def get_kaggle_competition_leaderboard(competition_slug: str):
    top_k = 10
    # fixme: doesn't work, cause of kaggle-api's bug <a href="https://github.com/Kaggle/kaggle-api/issues/622">https://github.com/Kaggle/kaggle-api/issues/622</a>
    command = ['kaggle', 'competitions', 'leaderboard', competition_slug, '--show']
    result = subprocess.run(command, stdout=subprocess.PIPE)
    print(result)

    leaderboard = result.stdout.decode('utf-8').strip().split('\n')
    data_lines = leaderboard[2:]
    print("data_lines: ", data_lines)
    leaderboard_data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 4:
            user_id = parts[0]
            username = ' '.join(parts[1:-2])
            timestamp = parts[-2]
            score = float(parts[-1])
            leaderboard_data.append((user_id, username, timestamp, score))
    leaderboard_topk = sorted(leaderboard_data, key=lambda x: x[3], reverse=True)[:top_k]
    print(leaderboard_topk)


def get_kaggle_competition_challenge(competition_slug: str):
    raise NotImplementedError


def get_kaggle_competition_topk_code(competition_slug: str, type: str):
    """
    get top k code of competition
    """
    top_k = 3
    command_to_topk_code = ['kaggle', 'kernels', 'list', '--competition', competition_slug, '--sort-by',
                            'scoreDescending', '--page-size', str(top_k)]
    result = subprocess.run(command_to_topk_code, stdout=subprocess.PIPE)
    output = result.stdout.decode()
    refs = re.findall(r'^([^\s]+)\s{2,}', output, re.MULTILINE)[2:]

    codes = []
    for ref in refs:
        competition_dir = KAGGLE_CASE_PATH / type / competition_slug
        command_get_jupyter = ['kaggle', 'kernels', 'pull', ref, '-p', str(competition_dir)]
        subprocess.run(command_get_jupyter, stdout=subprocess.PIPE)
        code = extract_code_from_notebook(competition_dir / (ref.split('/')[-1] + '.ipynb'))
        codes.append(code)

    logger.info("successfully get top k code of competition:" + competition_slug)
    return codes


def extract_code_from_notebook(notebook_path: str):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    code_cells = ""
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            code_cells += code + '\n'
    return code_cells


async def extract_exp_from_code(goal: str, code: str):
    # fixme: refactor extract_thought2 method & add_to_exp_bank method
    # fixed: design extract_thought_from_code function
    exp = await ThoughtExtract(config=gpt4t_config).extract_thought_from_code(goal, code)
    task_type = await QueryUtils().getQAType(goal, exp)
    # fixed: using add_to_exp_bank_with_metadata
    await add_to_exp_bank_with_metadata(goal, exp, task_type, EXP_PLAN)


async def extract_workflow_from_code(goal: str, code: str, exp: str = None):
    rsp = await ThoughtExtract(config=gpt4t_config).extract_workflow_from_code(goal, code)
    workflow = json.loads(rsp)
    if exp is not None:
        await add_to_workflow_exp_bank(workflow, exp, WORKFLOW_EXP)


# note: extract_exp_from_kaggle
async def extract_exp_from_kaggle():
    for types in kaggle_cases:
        logger.info("start to extract exp from kaggle: current type is: " + types['type'])
        for case in types['list']:
            goal = case['goal']
            codes = get_kaggle_competition_topk_code(case['slug'], types['type'])
            for code in codes:
                await extract_exp_from_code(goal, code)
            logger.success("Successfully extract exp from kaggle: " + case['slug'])


# note: addition to exp metadata
async def _fix_exp_bank_metadata():
    EXP_PLAN = EXAMPLE_DATA_PATH / "exp_bank/plan_exp.json"
    with open(EXP_PLAN, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        goal = item['task']
        exp = item['solution']
        task_type = await QueryUtils().getQAType(goal, exp)
        item['metadata'] = task_type
    with open(EXP_PLAN, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    logger.success("Successfully fix exp bank metadata")


# note: build workflow_exp_bank
async def build_workflow_exp_from_kaggle():
    kaggle_code_path = EXAMPLE_DATA_PATH / "kaggle_code"
    for types in kaggle_cases:
        logger.info("start to build workflow exp bank from kaggle: current type is: " + types['type'])
        task_type_path = kaggle_code_path / types['type']
        for case in types['list']:
            competition_path = task_type_path / case['slug']
            goal = case['goal']
            for file in os.listdir(competition_path):
                if file.endswith('.ipynb'):
                    code = extract_code_from_notebook(competition_path / file)

                    def get_exp_from_bank(task: str):
                        EXP_PLAN = EXAMPLE_DATA_PATH / "exp_bank/plan_exp.json"
                        with open(EXP_PLAN, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                        for item in data:
                            if item['task'] == task:
                                return item['solution']

                    await extract_workflow_from_code(goal, code, get_exp_from_bank(goal))
            logger.success("Successfully build workflow exp bank from kaggle code: " + case['slug'])


if __name__ == '__main__':
    # asyncio.run(extract_exp_from_kaggle())
    # asyncio.run(_fix_exp_bank_metadata())
    asyncio.run(build_workflow_exp_from_kaggle())
