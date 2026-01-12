import asyncio
from asyncio import WindowsSelectorEventLoopPolicy
from collections import Counter

from llama_index.core import QueryBundle
from scipy.optimize import linear_sum_assignment

from examples.ds_agent.ds_dataset_info import get_format_ds_question
from metagpt.actions.ds_agent.retrieval_exp import RetrievalExp
from metagpt.const import EXAMPLE_PATH, EXAMPLE_DATA_PATH
from metagpt.rag.engines.GraphMatching.graph import GraphSet
from metagpt.rag.engines.customMixture import CustomMixtureEngine
from metagpt.rag.schema import MixtureRetrieverConfig
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.strategy import ds_planner


workflow_ged_threshold_result = EXAMPLE_PATH / "ds_agent" / "experiment" / "workflow_ged_threshold_result.md"
EXP_PLAN = EXAMPLE_DATA_PATH / "exp_bank/plan_exp.json"


def _getGraphEditDistance(graph1: GraphSet, graph2: GraphSet):
    n, m = len(graph1.curVSet(0)), len(graph2.curVSet(0))
    matrix = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            dis = 0
            if graph1.curVSet(0).get(str(i + 1)) != graph2.curVSet(0).get(str(j + 1)):
                dis += 1
            g1_neighbors = graph1.neighbor_with_type(0, i)
            g2_neighbors = graph2.neighbor_with_type(0, j)
            g1_before, g1_after = g1_neighbors.get('before'), g1_neighbors.get('after')
            g2_before, g2_after = g2_neighbors.get('before'), g2_neighbors.get('after')

            def list_difference(list1, list2):
                count1 = Counter(list1)
                count2 = Counter(list2)
                difference = 0
                for element in count1:
                    diff = abs(count1[element] - count2.get(element, 0))
                    difference += diff
                for element in count2:
                    if element not in count1:
                        difference += count2[element]
                return difference

            matrix[i][j] = dis + list_difference(g1_before, g2_before) + list_difference(g1_after, g2_after)
    row_ind, col_ind = linear_sum_assignment(matrix)
    edit_distance = sum(matrix[i][j] for i, j in zip(row_ind, col_ind))
    return edit_distance


class ExperimentWorkflowGEDThreshold(Role):
    rag_engine: CustomMixtureEngine = None
    rag_exp_similarity_threshold: float = 0.4

    def init(self):
        self.rag_engine = CustomMixtureEngine.from_docs(
            input_files=[EXP_PLAN],
            retriever_configs=[MixtureRetrieverConfig()],
            ranker_configs=[],
        )

    async def plan(self, goal: str):
        pre_workflow : list[GraphSet] = []
        max_ged_threshold = -1
        ged_list = []
        for _ in range(3):
            self.planner = ds_planner.Planner(goal=goal, working_memory=self.rc.working_memory, auto_run=True)
            nodes_with_score = await self.rag_engine.aretrieve(goal)
            nodes_with_score = nodes_with_score[:1]
            if nodes_with_score[0].score > self.rag_exp_similarity_threshold:
                retrieval_res = await self.rag_engine.get_synthesizer_response(nodes_with_score, QueryBundle(goal))
                self.rc.working_memory.add(Message(content=retrieval_res.response, role="user", cause_by=RetrievalExp))
            await self.planner.update_plan(goal=goal)
            cur_graph = GraphSet(self.planner.plan)
            for pre_graph in pre_workflow:
                cur_ged = _getGraphEditDistance(cur_graph, pre_graph)
                ged_list.append(cur_ged)
                max_ged_threshold = max(max_ged_threshold, cur_ged)
            pre_workflow.append(cur_graph)
        with open(workflow_ged_threshold_result, "a") as f:
            f.write(f"{ged_list} -> {max_ged_threshold}\n")


if __name__ == '__main__':
    # qid_list = [0, 1, 2, 3, 4, 5, 6, 57, 181, 189]
    qid_list = [6]
    experiment = ExperimentWorkflowGEDThreshold()
    experiment.init()
    for qid in qid_list:
        question = get_format_ds_question(qid)
        asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
        asyncio.get_event_loop().run_until_complete(experiment.plan(goal=question))

