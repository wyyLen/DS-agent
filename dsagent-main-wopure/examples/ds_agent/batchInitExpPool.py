import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from metagpt.actions.ds_agent.extract_thought import ThoughtExtract
from metagpt.actions.ds_agent.query_utils import QueryUtils
from metagpt.actions.ds_agent.write_ds_plan import RefinePlan
from metagpt.config2 import Config
from metagpt.logs import logger
from metagpt.roles.ds_agent.ds_agent import add_to_exp_bank_with_metadata, add_to_workflow_exp_bank, \
    remove_last_item_from_exp_bank, EXP_PLAN
from metagpt.schema import Task, Plan

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.const import EXAMPLE_DATA_PATH, gpt4o_config_path, gpt4omini_config_path, WORKFLOW_EXP

gpt4o_config = Config.from_yaml_file(gpt4o_config_path)
gpt4omini_config = Config.from_yaml_file(gpt4omini_config_path)
KAGGLE_CASE_PATH = EXAMPLE_DATA_PATH / "kaggle_code"
keywords = [
    "data-analysis",
    "data-science",
    "data",
]
batch1_existed = {
    "commonlit-evaluate-student-summaries", "feedback-prize-2021", "playground-series-s3e11",
    "playground-series-s3e14", "python-and-analyze-data-final-project", "titanic"
}

current_existed_competitions_path = EXAMPLE_DATA_PATH / "exp_bank/info/exp_bank_current_competitions.json"
all_competitions_path = EXAMPLE_DATA_PATH / "exp_bank/info/kaggle_all_competitions.json"
all_competitions_config_path = EXAMPLE_DATA_PATH / "exp_bank/info/kaggle_all_competitions_config.md"
sampled_competitions_path = EXAMPLE_DATA_PATH / "exp_bank/info/sampled_competitions.json"


def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 无头模式，不显示浏览器
    options.add_argument('--disable-gpu')  # 禁用 GPU 加速，适合无头模式
    options.add_argument('--disable-dev-shm-usage')  # 防止内存不足问题
    return webdriver.Chrome(options=options)


def parse_page(html):
    # 解析页面中的 <div id='description'> 标签中的 <p> 文本
    soup = BeautifulSoup(html, 'html.parser')
    div = soup.find('div', id='description')
    if div:
        paragraphs = div.find_all('p')
        return [p.get_text() for p in paragraphs]
    else:
        print("No <div id='description'> found.")
        return []


class KaggleUtil:
    def count_for_all_competitions_with_keyword(self):
        # func: 收集 kaggle 中所有符合要求的竞赛
        # 初始化 all_competitions
        if os.path.exists(all_competitions_path):
            with open(all_competitions_path, 'r') as file:
                all_competitions = set(json.load(file))
        else:
            logger.error(f"File {all_competitions_path} not found")
            all_competitions = set()
        # 初始化 configs
        configs = set()
        if os.path.exists(all_competitions_config_path):
            try:
                with open(all_competitions_config_path, 'r') as file:
                    for line in file:
                        configs.add(line.strip())
            except Exception as e:
                logger.error(f"Error reading {all_competitions_config_path}: {e}")
        else:
            logger.error(f"File {all_competitions_config_path} not found")

        print(f"当前统计到的比赛数量：{len(all_competitions)}")
        for key in keywords:
            for page in range(40, 51):
                config_key = f"{key}@{page}"
                if config_key in configs:
                    logger.info(f"Skipping already processed config: {config_key}")
                    continue
                try:
                    competitions = self.get_past_competition(keyword=key, page=page)
                    logger.info(f"{config_key} 收集到{len(competitions)}个竞赛: {competitions}")
                    for c in competitions:
                        all_competitions.add(c)

                    if len(competitions) == 0:
                        continue

                    configs.add(config_key)

                    # 增量写入 all_competitions 到文件
                    try:
                        with open(all_competitions_path, 'w') as file:
                            json.dump(list(all_competitions), file, indent=4)
                    except Exception as e:
                        logger.error(f"Error writing to {all_competitions_path}: {e}")
                        raise  # 重新抛出异常，避免标记为已处理

                    # 标记该配置为已处理
                    configs.add(config_key)
                    try:
                        with open(all_competitions_config_path, 'a') as file:
                            file.write(f"{config_key}\n")
                    except Exception as e:
                        logger.error(f"Error writing to {all_competitions_config_path}: {e}")
                        raise  # 重新抛出异常，避免数据不一致

                except Exception as e:
                    logger.error(f"Error processing {config_key}: {e}")

    def get_past_competition(self, keyword: str = "data", page: int = 1):
        with create_driver() as driver:
            url = "https://www.kaggle.com/competitions?listOption=completed&sortOption=default&searchQuery=" + keyword
            if page > 1:
                url += "&page=" + str(page)
            driver.get(url)
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'MuiList-root'))
                )
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                competition_links = []
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.startswith("/competitions/"):
                        competition_links.append("https://www.kaggle.com" + href)
                return competition_links
            except Exception as e:
                print(f"Error: {e}")
                return None

    def get_competitions_list(self, keywords: list):
        # desc -> case:  kaggle competitions list -s data --sort-by numberOfTeams
        ref_set = set()
        for key in keywords:
            command_get_topk_competitions = ['kaggle', 'competitions', 'list', '-s', key, '--sort-by', 'numberOfTeams']
            result = subprocess.run(command_get_topk_competitions, stdout=subprocess.PIPE)
            output = result.stdout.decode()
            lines = output.splitlines()[2:]
            for line in lines:
                match = re.match(r'([^\s]+)\s{2,}(.+?)\s{2,}(.+?)\s{2,}(.+?)\s{2,}(.+?)\s{2,}(.+)', line)
                if match:
                    ref = match.group(1)
                    team_count = int(match.group(5))
                    if team_count > 300:
                        ref_set.add(ref)
        return ref_set

    def get_competition_description(self, url, max_retries=3, retry_delay=2):
        attempt = 0
        while attempt < max_retries:
            with create_driver() as driver:
                driver.get(url)
                try:
                    WebDriverWait(driver, 30).until(
                        EC.presence_of_element_located((By.ID, 'description'))
                    )
                    rsp = parse_page(driver.page_source)
                    description = ""
                    for r in rsp:
                        description += r + "\n"
                    if description.strip():  # 检查description是否为空
                        return description
                    else:
                        print(f"Attempt {attempt + 1}: Description is empty, retrying...")
                        time.sleep(retry_delay)
                except Exception as e:
                    print(f"Error on attempt {attempt + 1} to get description: {e}")
                    time.sleep(retry_delay)
            attempt += 1
        print(f"Failed after {max_retries} attempts to get description.")
        return None

    def get_competition_files(self, slug: str, max_retries=3, retry_delay=2):
        # desc -> case: kaggle competitions files <slug>
        attempt = 0
        while attempt < max_retries:
            command_get_competition_files = ['kaggle', 'competitions', 'files', slug]
            try:
                result = subprocess.run(command_get_competition_files, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output = result.stdout.decode()
                if result.returncode == 0:
                    file_names = re.findall(r'([a-zA-Z_]+\.\w+)', output)
                    if file_names:  # 如果找到了文件名则返回
                        return file_names
                    else:
                        print(f"Attempt {attempt + 1}: No file names found, retrying...")
                else:
                    print(f"Attempt {attempt + 1}: Error in subprocess to get files, retrying...")
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")

            time.sleep(retry_delay)
            attempt += 1

        print(f"Failed after {max_retries} attempts to get files.")
        return []

    def get_kaggle_competition_topk_code(self, competition_slug: str, type: str, top_k: str = 3, max_retries=3, retry_delay=2):
        attempt = 0
        codes, refs = [], []

        while attempt < max_retries:
            # 获取TopK代码的命令
            command_to_topk_code = ['kaggle', 'kernels', 'list', '--competition', competition_slug, '--sort-by',
                                    'voteCount', '--page-size', str(top_k)]
            try:
                result = subprocess.run(command_to_topk_code, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output = result.stdout.decode()
                refs = re.findall(r'^([^\s]+)\s{2,}', output, re.MULTILINE)[2:]
                if refs:
                    break  # 如果成功获取到TopK代码，则跳出重试循环
                else:
                    print(f"Attempt {attempt + 1}: No TopK references found, retrying...")
            except Exception as e:
                return []

            time.sleep(retry_delay)
            attempt += 1

        if not refs:
            print(f"Failed to get TopK references after {max_retries} attempts.")
            return codes

        # 获取代码的命令
        competition_dir = Path(KAGGLE_CASE_PATH) / type / competition_slug
        for ref in refs:
            attempt_code = 0
            while attempt_code < max_retries:
                command_get_jupyter = ['kaggle', 'kernels', 'pull', ref, '-p', str(competition_dir)]
                try:
                    result_code = subprocess.run(command_get_jupyter, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    file_path = competition_dir / ref.split('/')[-1]
                    try:
                        code = extract_code_from_notebook(file_path.with_suffix('.ipynb'))
                        codes.append(code)
                        break
                    except FileNotFoundError:
                        try:
                            code = extract_code_from_py(file_path.with_suffix('.py'))
                            codes.append(code)
                            break
                        except FileNotFoundError:
                            print(f"Attempt {attempt_code + 1}: Error in pulling kernel, retrying...")
                except Exception as e:
                    print(f"Error on attempt {attempt_code + 1}: {e}")

                time.sleep(retry_delay)
                attempt_code += 1

            if attempt_code == max_retries:
                print(f"Failed to get code for {ref} after {max_retries} attempts.")
                continue  # 即使获取某个代码失败，依然继续处理其他的TopK代码

        logger.info("Successfully got top K code of competition: " + competition_slug)
        return codes


async def batchGotExpMain(keywords: list):
    competitions_set = KaggleUtil().get_competitions_list(keywords)
    competitions = list(competitions_set)
    await _gotExpCompetitions(competitions)


async def batchGotExpMainForPastComp(keyword: str = "data", page: int = 1):
    competitions = KaggleUtil().get_past_competition(keyword, page)
    print(f"current_got_competitions: {len(competitions)}")
    await _gotExpCompetitions(competitions)


async def batchGotSupplementComp():
    if os.path.exists(sampled_competitions_path):
        with open(sampled_competitions_path, 'r') as file:
            sampled_competitions_set = set(json.load(file))
    else:
        logger.error(f"File {sampled_competitions_path} not found")
        sampled_competitions_set = set()
    print(f"To be selected competitions: {sampled_competitions_set}")
    competitions = list(sampled_competitions_set)
    await _gotExpCompetitions(competitions)


async def _gotExpCompetitions(competitions: list):
    if os.path.exists(current_existed_competitions_path):
        with open(current_existed_competitions_path, 'r') as file:
            existing_competitions = set(json.load(file))
    else:
        logger.error(f"File {current_existed_competitions_path} not found")
        existing_competitions = set()
    print(f"existing_competitions: {len(existing_competitions)}")
    new_competitions = [comp for comp in competitions if comp not in existing_competitions]
    print(f"new_competitions: {len(new_competitions)}")
    print(f"new_competitions detail: {new_competitions}")

    for competition_ref in new_competitions:
        slug = competition_ref.split('/')[-1]
        if slug in batch1_existed:
            continue
        comp_desc = KaggleUtil().get_competition_description(competition_ref)
        if comp_desc is None or len(comp_desc) == 0:
            continue
        files = KaggleUtil().get_competition_files(slug)
        goal = await _extract_goal_from_competition_description(comp_desc, files)
        codes = KaggleUtil().get_kaggle_competition_topk_code(slug, "batch2")
        if len(codes) <= 1:
            continue
        for code in codes:
            exp = await _extract_exp_from_code(goal, code)
            try:
                await _extract_workflow_from_code(goal, code, exp)
            except Exception as e:
                logger.error("error appeared when extract_workflow_from_code")
                remove_last_item_from_exp_bank(EXP_PLAN)

        existing_competitions.add(competition_ref)
        with open(current_existed_competitions_path, 'w') as file:
            json.dump(list(existing_competitions), file, indent=4)
        logger.success(f"Successfully got new exp from {competition_ref}")


async def _getExpFromCodeWithGoal(goal: str, slug: str):
    # desc 作为上述_gotExpCompetitions方法异常失败后的备选方案，补全剩余的代码的解决方案
    raise NotImplementedError


async def _extract_goal_from_competition_description(description: str, files: list):
    goal = await ThoughtExtract(config=gpt4omini_config).extract_goal_from_competition_description(description, files)
    return goal


async def _extract_workflow_from_code(goal: str, code: str, exp: str = None):
    retry_count, task_list = 0, []
    while retry_count < 3:
        try:
            rsp = await ThoughtExtract(config=gpt4o_config).extract_workflow_from_code(goal, code)
            workflow = json.loads(rsp)
            task_list = [Task(**task) for task in workflow]
            break
        except Exception as e:
            logger.warning(f"Failed to extract workflow from code: {e}")
            retry_count += 1
    if retry_count == 3:
        logger.error(f"Failed to extract workflow from code after 3 retries: {code}")
        raise Exception
    mock_plan = Plan(goal=goal)
    mock_plan.tasks = task_list
    await RefinePlan(config=gpt4o_config).refine_ds_scenarios_in_plan(mock_plan)
    if exp is not None:
        await add_to_workflow_exp_bank(mock_plan, exp, WORKFLOW_EXP)


async def _extract_exp_from_code(goal: str, code: str):
    exp = await ThoughtExtract(config=gpt4o_config).extract_thought_from_code(goal, code)
    task_type = await QueryUtils(config=gpt4o_config).getQAType(goal, exp)
    await add_to_exp_bank_with_metadata(goal, exp, task_type, EXP_PLAN)
    return exp


async def get_ref_jupyter(ref: str, competition_dir) -> str:
    command_get_jupyter = ['kaggle', 'kernels', 'pull', ref, '-p', str(competition_dir)]
    subprocess.run(command_get_jupyter, stdout=subprocess.PIPE)
    code = extract_code_from_notebook(competition_dir / (ref.split('/')[-1] + '.ipynb'))
    return code


def extract_code_from_notebook(notebook_path: str) -> str:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    code_cells = ""
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            code_cells += code + '\n'
    return code_cells


def extract_code_from_py(py_file_path: str) -> str:
    with open(py_file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    return code


async def _try():
    # competitions = _get_competitions_list(keywords)
    comp_ref = "https://www.kaggle.com/competitions/datamix-data-science-course-nlp"
    # comp_ref = "https://www.kaggle.com/competitions/titanic"
    slug = comp_ref.split('/')[-1]
    comp_files = KaggleUtil().get_competition_files(slug)
    comp_desc = KaggleUtil().get_competition_description(comp_ref)
    print(comp_ref)
    print(comp_desc)
    goal = await _extract_goal_from_competition_description(comp_desc, comp_files)
    print(goal)


def func(key):
    # desc 演示 通过关键词的竞赛批量获取
    command_get_topk_competitions = ['kaggle', 'competitions', 'list', '-s', key, '--sort-by', 'numberOfTeams']
    result = subprocess.run(command_get_topk_competitions, stdout=subprocess.PIPE)
    output = result.stdout.decode()
    return output


def count_cur_exps():
    with open(current_existed_competitions_path, 'r', encoding='utf-8') as file:
        current_existed_competitions = json.load(file)
        print(f"current_existed_competitions count: {len(current_existed_competitions)}")
    with open(EXP_PLAN, 'r', encoding='utf-8') as file:
        exp_data = json.load(file)
        print(f"textualize exp count: {len(exp_data)}")
    with open(WORKFLOW_EXP, 'r', encoding='utf-8') as file:
        workflow_data = json.load(file)
        print(f"textualize exp count: {len(workflow_data)}")


def check_for_duplicated():
    with open(sampled_competitions_path, 'r', encoding='utf-8') as file:
        sampled_competitions_set = json.load(file)
    s = set()
    for comp in sampled_competitions_set:
        if comp not in s:
            s.add(comp)
        else:
            print(f"{comp} is duplicated")


if __name__ == '__main__':
    # desc 使用batchGotExpMain获取当前正在进行的竞赛，基于kaggle api实现。 func 参数 keywords（搜索关键词）
    # desc 使用batchGotExpMainForPastComp获取已经结束的竞赛，基于webdriver自定义实现。 func 参数 keyword（关键词），page（页码）
    # asyncio.run(batchGotExpMainForPastComp(keyword="data+analysis", page=2))
    # asyncio.run(batchGotExpMain(keywords))
    asyncio.run(batchGotSupplementComp())

    # past_competitions = get_past_competition(keyword=keywords[2], page=1)
    # print(past_competitions)
    # print(json.dumps(past_competitions))
    # print(len(past_competitions))
    pass
    # count_for_all_competitions()    # desc 收集 kaggle 中所有符合要求的竞赛
