import asyncio
import json
import logging
import os
import sys

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.rag.engines import CustomEngine
from metagpt.rag.schema import FAISSRetrieverConfig
from metagpt.const import EXAMPLE_DATA_PATH, DATA_PATH


EXP_PATH = EXAMPLE_DATA_PATH / "exp_bank/plan_exp.json"
DOC_PATH = EXAMPLE_DATA_PATH / "rag/1.txt"
JSON_PATH = EXAMPLE_DATA_PATH / "search_kb/plan_exp.json"
question_file = os.path.join(DATA_PATH, "di_dataset/da_bench/da-dev-questions.jsonl")

PROMPT_IN_PAPER = """
File: {file_path}
Question: {question}
Constraints: {constraints}
"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def get_question():
    with open(question_file, 'r') as f:
        data = f.readlines()
    for i, line in enumerate(data):
        if i != 54:
            continue
        item = json.loads(line)
        question = item['question']
        constraints = item['constraints']
        file_name = item['file_name']
        requirement = PROMPT_IN_PAPER.format(
            file_path=os.path.abspath(os.path.join(DATA_PATH, "di_dataset/da_bench/da-dev-tables", file_name)),
            question=question, constraints=constraints)
        return requirement


async def main():
    question = get_question()
    print("current question: ", question)

    engine = CustomEngine.from_docs(input_files=[EXP_PATH], retriever_configs=[FAISSRetrieverConfig()])
    retrieval_res = await engine.aquery(question)
    logging.info(f"retrieval_res: {retrieval_res}")


if __name__ == "__main__":
    asyncio.run(main())


"""
{
'2778f783-f065-48df-9ffe-0198cce9a507': RefDocInfo(
    node_ids=['be14b6a9-837c-45ea-b7e6-35f4bae62f3f'], 
    metadata={
        'file_path': 'D:\\Dev\\DSAgent\\examples\\data\\search_kb\\plan_exp.json', 
        'file_name': 'D:/Dev/DSAgent/examples/data/search_kb/plan_exp.json', 
        'file_type': 'application/json', 
        'file_size': 2268, 
        'creation_date': 
        '2024-07-13', 'last_modified_date': '2024-07-13'}
    )}


关键字表索引：
query keywords: ['height', 'length', 'feature', 'diameter', 'weight', 'volume', 'abalone', 'set', 'linear', 'test']
Extracted keywords: ['length', 'feature', 'abalone', 'set']
"""
