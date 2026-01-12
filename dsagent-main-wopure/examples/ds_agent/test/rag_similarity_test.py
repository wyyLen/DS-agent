from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

from examples.ds_agent.ds_dataset_info import get_format_ds_question
from metagpt.logs import logger
from metagpt.roles.ds_agent.ds_agent import DSAgent

ds = DSAgent(use_reflection=True)
engine = ds.rag_engine
index = engine.get_index()
retriever = engine.get_retriever()
vector_store = index.vector_store
# print("similarity_top_k:", retriever.similarity_top_k())
# print("if the vector store is_embedding_query:", retriever.get_vector_store().is_embedding_query)


# question = """
# File: D:\Dev\DSAgent\data\di_dataset\da_bench\da-dev-tables\abalone.csv
# Question: Explore the correlation between the length and the weight of the whole abalone. Additionally, perform feature engineering by creating a new feature called "volume" by multiplying the length, diameter, and height of the abalone. Determine if the volume feature improves the accuracy of predicting the number of rings using a linear regression model.
# Constraints: Calculate the Pearson correlation coefficient to assess the strength and direction of the linear relationship between length and the weight. The volume feature should be created by multiplying the length, diameter, and height of the abalone. Use the sklearn's linear regression model to predict the number of rings. Split the data into a 70% train set and a 30% test set. Evaluate the models by calculating the root mean squared error (RMSE) with the test set.
# """


async def main(q_id: int):
    query = get_format_ds_question(q_id)
    # print("current question: ", query)
    # nodes_with_score, query_bundle = await engine.aquery(question)
    nodes_with_score = await engine.aretrieve(query)   # 等价于 engine.aquery(question)  query_bundle 可以通过QueryBundle(query)获取
    if nodes_with_score is None:
        logger.info("No nodes found")
        return

    for node in nodes_with_score:
        if not isinstance(node, NodeWithScore):
            raise TypeError("Node must be a NodeWithScore object")

        # note: 获取 NodeWithScore 中节点的 id_、文本和相似度分数
        d = node.to_dict()
        # print("id:", d.get("node").get("id_"))
        # print(d.get("node").get("text"))
        print("score:", d.get("score"))  # or node.get_score()


if __name__ == '__main__':
    import asyncio
    for i in range(10, 40):
        asyncio.run(main(i))



