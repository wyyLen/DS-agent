import json

from dsagent_core.const import EXAMPLE_DATA_PATH
from dsagent_core.rag.engines.GraphMatching.graph import GraphSet
from dsagent_core.rag.engines.graphUtils import _json2plan, _getGraphEditDistance
from metagpt.schema import Plan

WORKFLOW_EXP_PATH = EXAMPLE_DATA_PATH / "exp_bank/workflow_exp2.json"
WORKFLOW_EXP_PATH_CLEAN = EXAMPLE_DATA_PATH / "exp_bank/workflow_exp2_clean.json"


class CustomWorkflowGMEngine:
    workflow_exp_bank: dict[GraphSet, str]
    workflow_ged_threshold: int

    def __init__(self, workflow_exp_path):
        self.workflow_exp_bank = {}
        with open(workflow_exp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            cur_plan = _json2plan(item["workflow"])
            graph = GraphSet(cur_plan)
            self.workflow_exp_bank[graph] = item["exp"]
        self.workflow_ged_threshold = 12

    def retrieval(self, workflow: Plan):
        cur_graph = GraphSet(workflow)
        most_similar_graphs, similarity = None, 999
        for graph in self.workflow_exp_bank.keys():
            edit_distance = _getGraphEditDistance(graph, cur_graph)
            # print(edit_distance)
            if edit_distance < self.workflow_ged_threshold and edit_distance < similarity:
                most_similar_graphs = graph
                similarity = edit_distance

        if not most_similar_graphs:
            return None

        return self.workflow_exp_bank.get(most_similar_graphs)

    def add(self, workflow: Plan, exp: str):
        graph = GraphSet(workflow)
        self.workflow_exp_bank.setdefault(graph, exp)


# def main():
#     workflow_engine = CustomWorkflowGMEngine(workflow_exp_path=WORKFLOW_EXP_PATH_CLEAN)
#     plan1 = get_fixed_plan(1812)
#     res = workflow_engine.retrieval(plan1)
#     print(res)
#
#
# main()
