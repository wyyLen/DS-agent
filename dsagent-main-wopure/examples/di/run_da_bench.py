import os
import fire
import sys
import json

import yaml

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.const import DATA_PATH
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.tools.tool_recommend import TypeMatchToolRecommender
from openai import OpenAI

question_file = os.path.join(DATA_PATH, "di_dataset/da_bench/da-dev-questions.jsonl")

PROMPT_WITH_WORKFLOW = """
File: {file_path}
Question: {question}
Constraints: {constraints}
Workflow: {workflow}
"""

workflow181 = """
1. Perform exploratory data analysis (EDA) on the abalone dataset to understand the data distribution and relationships.
2. Explore the correlation between the length and the weight of the whole abalone.
3. Perform feature engineering by creating a new feature 'volume' by multiplying the length, diameter, and height of the abalone.
4. Using the new feature 'volume, a linear regression model is trained on the dataset using sklearn to predict the number of rings.
5. Without using the new feature 'volume, use sklearn to train a linear regression model on the dataset to predict the number of rings.
6. Evaluate the trained linear regression model by calculating the root mean squared error (RMSE) using the test set.
"""

workflow_test = """
The following should be noted:
1. For the correlation, first determine the correlation coefficient used and the correlation coefficient detection standard, and finally output the correlation analysis results.
2. To determine whether a certain feature has an impact on the prediction accuracy of the model, the model training and prediction should be performed separately for the two cases of using this feature and not using this feature.
"""

PROMPT_IN_PAPER = """
File: {file_path}
Question: {question}
Constraints: {constraints}
"""


async def ask_gpt(data_dir=DATA_PATH):
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/da_bench")):
        raise FileNotFoundError(f"da_bench dataset not found in {data_dir}.")

    config_path = os.path.join("C:\\Dev\\project\\github-project\\MetaGPT", "config/config2.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config not found in {config_path}.")

    with open(question_file, 'r') as f:
        data = f.readlines()

    for i, line in enumerate(data):
        if i != 3:
            continue
        item = json.loads(line)
        question = item['question']
        constraints = item['constraints']
        file_name = item['file_name']
        requirement = PROMPT_IN_PAPER.format(
            file_path=os.path.abspath(os.path.join(data_dir, "di_dataset/da_bench/da-dev-tables", file_name)),
            question=question, constraints=constraints)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        client = OpenAI(api_key=config['llm']['api_key'])
        rsp = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": requirement}],
        )
        print(rsp.choices[0].message.content)


async def main(data_dir=DATA_PATH, use_reflection=True):
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/da_bench")):
        raise FileNotFoundError(f"da_bench dataset not found in {data_dir}.")

    with open(question_file, 'r') as f:
        data = f.readlines()

    for i, line in enumerate(data):
        """
        181: Lack of planning skills 缺乏规划能力
        
        """
        if i != 181:
            continue
        item = json.loads(line)
        question = item['question']
        constraints = item['constraints']
        concepts = item['concepts']
        answer_format = item['format']
        file_name = item['file_name']
        requirement = PROMPT_IN_PAPER.format(
            file_path=os.path.abspath(os.path.join(data_dir, "di_dataset/da_bench/da-dev-tables", file_name)),
            question=question, constraints=constraints)
        # requirement = PROMPT_WITH_WORKFLOW.format(
        #     file_path=os.path.abspath(os.path.join(data_dir, "di_dataset/da_bench/da-dev-tables", file_name)),
        #     question=question, constraints=constraints, workflow=workflow_test)
        print(requirement)
        di = DataInterpreter(use_reflection=use_reflection, tool_recommender=TypeMatchToolRecommender(tools=[]))
        await di.run(requirement)


async def get_dataset_outline(data_dir=DATA_PATH):
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/da_bench")):
        raise FileNotFoundError(f"da_bench dataset not found in {data_dir}.")
    with open(question_file, 'r') as f:
        data = f.readlines()
    for i, line in enumerate(data):
        item = json.loads(line)
        if item['level'] == "hard":
            print("idx:{idx}, concept:{concept}, level:{level}"
                  .format(concept=item['concepts'], level=item['level'], idx=i))


if __name__ == "__main__":
    fire.Fire(main)
