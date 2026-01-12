import asyncio
import json
import os
import sys
from asyncio import WindowsSelectorEventLoopPolicy
from typing import List, Union

import fire

from examples.ds_agent.ds_dataset_info import get_format_ds_question
from examples.ds_agent.reformat import reformat
from examples.experiment.da_bench.util.DABENCH import DABench
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.roles.ds_agent.ds_agent_stream import DSAgentStream
from metagpt.tools.tool_recommend import TypeMatchToolRecommender

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.const import DATA_PATH, CUSTOM_DA_EVAL_RES_FILE, DI_EVAL_RES_FILE
from metagpt.roles.ds_agent.ds_agent import DSAgent

question_file = os.path.join(DATA_PATH, "di_dataset/da_bench/da-dev-questions.jsonl")

PROMPT_IN_PAPER = """
File: {file_path}
Question: {question}
Constraints: {constraints}
"""


# async def ask_gpt(data_dir=DATA_PATH):
#     if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/da_bench")):
#         raise FileNotFoundError(f"da_bench dataset not found in {data_dir}.")
#
#     config_path = os.path.join("C:\\Dev\\project\\github-project\\MetaGPT", "config/config2.yaml")
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"config not found in {config_path}.")
#
#     with open(question_file, 'r') as f:
#         data = f.readlines()
#
#     for i, line in enumerate(data):
#         if i != 3:
#             continue
#         item = json.loads(line)
#         question = item['question']
#         constraints = item['constraints']
#         file_name = item['file_name']
#         requirement = PROMPT_IN_PAPER.format(
#             file_path=os.path.abspath(os.path.join(data_dir, "di_dataset/da_bench/da-dev-tables", file_name)),
#             question=question, constraints=constraints)
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
#         client = OpenAI(api_key=config['llm']['api_key'])
#         rsp = client.chat.completions.create(
#             model="gpt-3.5-turbo-0125",
#             messages=[{"role": "user", "content": requirement}],
#         )
#         print(rsp.choices[0].message.content)


def init_ds_agent(use_rag: bool = True) -> DSAgent:
    return DSAgent(use_reflection=True, use_rag=use_rag, use_kaggle_exp=True, use_exp_extractor=False)

def init_stream_ds_agent(use_rag: bool = True) -> DSAgentStream:
    return DSAgentStream(use_reflection=True, use_rag=use_rag, use_kaggle_exp=True, use_exp_extractor=False)


def get_questions(data_dir: str) -> List[str]:
    # if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/da_bench")):
    #     raise FileNotFoundError(f"da_bench dataset not found in {data_dir}.")
    # with open(question_file, 'r') as f:
    #     data = f.readlines()
    questions = []
    bench = DABench()
    prompt = bench.generate_formatted_prompt(549)
    print(f"question: {prompt}")
    # print(f"question: {prompt} \n answer: {bench.get_answer(0)}")
    questions.append(prompt)
    # for i, line in enumerate(data):
    #     """
    #     181: Lack of planning skills 缺乏规划能力
    #     57: 相关性系数计算有问题，需要结合ds_exp
    #     189: 数据预处理+相关性分析
    #     """s
    #     if i != 181:
    #         continue
    #     item = json.loads(line)
    #     cur_question = item['question']
    #     constraints = item['constraints']
    #     file_name = item['file_name']
    #     requirement = PROMPT_IN_PAPER.format(
    #         file_path=os.path.abspath(os.path.join(data_dir, "di_dataset/da_bench/da-dev-tables", file_name)),
    #         question=cur_question, constraints=constraints)
    #     questions.append(requirement)
    return questions


async def run_ds_agent(agent: Union[DSAgent, DSAgentStream], requirement: str):
    if agent is None:
        raise ValueError("Agent is not initialized.")
    print(requirement)
    rsp = await agent.run(requirement)
    return rsp.content


async def run_data_interpreter(requirement: str, di: DataInterpreter = None):
    if di is None:
        di = DataInterpreter(use_reflection=True, tool_recommender=TypeMatchToolRecommender(tools=[]))
    print(requirement)
    rsp = await di.run(requirement)
    return rsp.content


async def run_our_agent_da_bench():
    if not os.path.exists(os.path.join(DATA_PATH, "di_dataset", "da_bench")):
        raise FileNotFoundError(f"da_bench dataset not found in {DATA_PATH}.")
    with open(question_file, 'r') as f:
        questions = f.readlines()
    dsAgent = init_ds_agent()
    for i, line in enumerate(questions):
        if i <= 10 or i >= 20:
            continue
        item = json.loads(line)
        requirement = PROMPT_IN_PAPER.format(
            file_path=os.path.abspath(os.path.join(DATA_PATH, "di_dataset/da_bench/da-dev-tables", item['file_name'])),
            question=item['question'], constraints=item['constraints'])
        rsp = await run_ds_agent(dsAgent, requirement)
        print("+++++++++++++++++++", rsp)
        reformat_and_eval = reformat(rsp, line)
        with open(CUSTOM_DA_EVAL_RES_FILE, 'a', encoding='utf-8') as file:
            file.write(str(reformat_and_eval) + '\n')
        dsAgent.clear_content()


async def run_data_interpreter_da_bench():
    if not os.path.exists(os.path.join(DATA_PATH, "di_dataset", "da_bench")):
        raise FileNotFoundError(f"da_bench dataset not found in {DATA_PATH}.")
    with open(question_file, 'r') as f:
        questions = f.readlines()
    for i, line in enumerate(questions):
        # if i <= 2 or i >= 10:
        #     continue
        if i != 10:
            continue
        item = json.loads(line)
        requirement = PROMPT_IN_PAPER.format(
            file_path=os.path.abspath(os.path.join(DATA_PATH, "di_dataset/da_bench/da-dev-tables", item['file_name'])),
            question=item['question'], constraints=item['constraints'])
        di = DataInterpreter(use_reflection=True, tool_recommender=TypeMatchToolRecommender(tools=[]))
        try:
            rsp = await run_data_interpreter(requirement, di)
            print("+++++++++++++++++++", rsp)
            reformat_and_eval = reformat(rsp, line)
        except Exception as e:
            reformat_and_eval = {
                "id": item['id'],
                "label_answers": {
                    "all": "0"
                },
                "predicted_answers": {
                    "all": "1"
                },
                "correctness": {"all": False}
            }
            print(e)
        with open(DI_EVAL_RES_FILE, 'a', encoding='utf-8') as file:
            file.write(str(reformat_and_eval) + '\n')


if __name__ == "__main__":
    # desc: ds-agent benchmark
    # asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    # asyncio.get_event_loop().run_until_complete(run_our_agent_da_bench())

    # desc: dataInterpreter benchmark
    # asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    # asyncio.get_event_loop().run_until_complete(run_data_interpreter_da_bench())

    # desc: single question for test
    question_list = get_questions(DATA_PATH)
    # ds = init_ds_agent()
    ds_stream = init_stream_ds_agent()
    asyncio.run(run_ds_agent(ds_stream, question_list[0]))
    # ds.clear_content()

    # desc: 与 dataInterpreter 对比
    # asyncio.run(run_data_interpreter(question_list[0]))
