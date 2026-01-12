import asyncio
import json
import logging
import os
import sys

from llama_index.core import Document

from metagpt.actions.ds_agent.query_utils import QueryUtils
from metagpt.rag.engines.customMixture import CustomMixtureEngine

sys.path.append('C:\\Dev\\project\\github-project\\MetaGPT')

from metagpt.rag.engines import CustomEngine
from metagpt.rag.schema import FAISSRetrieverConfig, MixtureRetrieverConfig
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


async def get_question_type():
    question = get_question()
    res = await QueryUtils().getQuestionType(question)
    print(res)
    print(res[0])


async def main():
    question = get_question()
    print("current question: ", question)
    print("-----------ready to build custom mixture engine---------------")
    engine = CustomMixtureEngine.from_docs(input_files=[EXP_PATH], retriever_configs=[MixtureRetrieverConfig()])
    print("-----------custom mixture engine has been built successfully---------------")
    retrieval_res = await engine.aquery(question)
    logging.info(f"retrieval_res: {retrieval_res}")


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(get_question_type())
