from llama_index.core import Document

from examples.ds_agent.ds_dataset_info import get_format_ds_question
from metagpt.rag.factories import get_retriever
from metagpt.rag.schema import FAISSRetrieverConfig
from dsagent_core.roles.ds_agent import DSAgent

ds = DSAgent(use_reflection=True)

new_docs = """
{
    "task": "\nFile: D:\\Dev\\DSAgent\\data\\di_dataset\\da_bench\\da-dev-tables\\test_ave.csv\nQuestion: Calculate the mean fare paid by the passengers.\nConstraints: Calculate the mean fare using Python's built-in statistics module or appropriate statistical method in pandas. Rounding off the answer to two decimal places.\n",
    "plan_output": "(1) **Summary of the Overall Design of the Plan:**\n   The plan for solving the data analysis problem involves two main tasks:\n   - **Task 1:** Load the dataset from a specified file ('test_ave.csv'). This is the preliminary step necessary to access the data which will be used for further analysis.\n   - **Task 2:** Calculate the mean fare paid by the passengers. This task depends on the successful completion of Task 1, as it requires the data to be loaded before any calculations can be performed.\n\n   The tasks are structured sequentially, where the output of the first task (loading the data) serves as the input for the second task (calculating the mean fare).\n\n(2) **Explanation of the Dependencies Between the Tasks:**\n   - **Task 1** has no dependencies. It is the initial step and involves loading the dataset from a CSV file. This task must be completed successfully to provide the dataset for any subsequent analysis.\n   - **Task 2** is dependent on **Task 1**. The dependency is logical and necessary because the calculation of the mean fare cannot proceed without the dataset being loaded first. The instruction to calculate the mean fare explicitly requires the dataset, which is made available by completing Task 1.\n\n   The dependency is indicated in the plan by the \"dependent_task_ids\" field for Task 2, which lists Task 1. This signifies that Task 2 can only be executed after the successful completion of Task 1.\n\n(3) **Pattern of Questions in the Current Problem Based on Multiple Steps in the Plan:**\n   The current problem, which is to calculate the mean fare paid by the passengers, inherently requires a multi-step approach as outlined in the plan:\n   - **Step 1:** Access the data by loading it from a file. This is a preparatory step that ensures the data needed for analysis is available.\n   - **Step 2:** Perform the specific analysis, which in this case is calculating the mean fare. This step is directly related to the question posed in the problem.\n\n   This pattern of having a data preparation/loading step followed by a data analysis/calculation step is common in data science workflows. It ensures that the data is properly prepared and available for any analytical operations needed to answer specific questions. The structured approach in the plan facilitates a clear and organized way to address the problem, ensuring that each step is logically sequenced to support the subsequent analysis."
}
"""


async def dynamic_add_exp(q_id: int):
    """
        this func allow to add new docs to the index, without reloading the whole rag engine.
    """
    index = ds.rag_engine.get_index()
    # doc_chunk = Document(text=new_docs)
    # index.insert(doc_chunk)
    # ds.rag_engine.update_rag_engine(index)
    ds.rag_engine.add_exp(new_docs)
    print(index.ref_doc_info)
    query = get_format_ds_question(q_id)
    nodes_with_score = await ds.rag_engine.aretrieve(query)
    for node in nodes_with_score:
        d = node.to_dict()
        print("score:", d.get("score"))


if __name__ == '__main__':
    import asyncio
    asyncio.run(dynamic_add_exp(181))
