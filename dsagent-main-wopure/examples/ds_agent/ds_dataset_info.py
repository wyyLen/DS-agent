import json
import os

from metagpt.const import DATA_PATH

question_file = os.path.join(DATA_PATH, "di_dataset/da_bench/da-dev-questions.jsonl")

PROMPT_IN_PAPER = """
File: {file_path}
Question: {question}
Constraints: {constraints}
"""


def get_format_ds_question(idx: int):
    data_dir = DATA_PATH
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/da_bench")):
        raise FileNotFoundError(f"da_bench dataset not found in {data_dir}.")
    with open(question_file, 'r') as f:
        data = f.readlines()

    item = json.loads(data[idx])
    requirement = PROMPT_IN_PAPER.format(
        file_path=os.path.abspath(os.path.join(data_dir, "di_dataset/da_bench/da-dev-tables", item['file_name'])),
        question=item['question'], constraints=item['constraints'])
    return requirement


def get_dataset_outline(data_dir=DATA_PATH):
    if data_dir != DATA_PATH and not os.path.exists(os.path.join(data_dir, "di_dataset/da_bench")):
        raise FileNotFoundError(f"da_bench dataset not found in {data_dir}.")
    with open(question_file, 'r') as f:
        data = f.readlines()
    for i, line in enumerate(data):
        item = json.loads(line)
        if item['level'] == "hard":
            print("idx:{idx}, concept:{concept}, level:{level}"
                  .format(concept=item['concepts'], level=item['level'], idx=i))

