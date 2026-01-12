import networkx as nx
from networkx import DiGraph

from dsagent_core.actions.ds_agent.fixed_plan_for_test import get_fixed_plan
from metagpt.logs import logger
from metagpt.schema import Plan, Task


class Node:
    pos: str
    type: str
    instruction: str

    def __init__(self, pos, type, instruction=None):
        self.pos = pos
        self.type = type
        self.instruction = instruction


class GraphSet:
    # todo: 当前只实现了单个任务计划的初始化，后续需要支持基于经验池或任务计划列表的初始化
    def __init__(self, plan: Plan):
        networkx_graph = _extract_graph_from_plan(plan)
        self.__graphSet = []     # [{'1':'pda', '2':'correlation analysis', '3':'feature engineering'}]
        self.__vertexSet = []    # [{'1':'pda', '2':'correlation analysis', '3':'feature engineering'}]
        self.__edgeSet = []      # [{'1:2': 1, '1:3': 1, '1:4': 1, '3:4': 1, '4:5': 1, '4:6': 1}]
        curVertexSet = {}     # 节点集合
        curEdgeSet = {}       # 边集合
        for node, attrs in networkx_graph.nodes(data=True):
            curVertexSet[node] = attrs    # update: 更新之后的节点内容包括了 attrs['task_type']和 attrs['instruction']
        for edge in networkx_graph.edges():
            edgeKey = str(edge[0]) + ":" + str(edge[1])
            curEdgeSet[edgeKey] = 1
        self.__graphSet.append(curVertexSet)
        self.__vertexSet.append(curVertexSet)
        self.__edgeSet.append(curEdgeSet)

    def graphSet(self):
        return self.__graphSet

    def curVSet(self, offset):
        if offset >= len(self.__vertexSet):
            logger.error("Class GraphSet curVSet() offset out of index!")
            exit()
        return self.__vertexSet[offset]

    # note: 在目前的实现中，offset的值基本保持为0。因为当前一个GraphSet本质上就是一个图。
    def curESet(self, offset):
        if offset >= len(self.__edgeSet):
            logger.error("Class GraphSet curESet() offset out of index!")
            exit()
        return self.__edgeSet[offset]

    def curVESet(self, offset) -> list:
        # desc: 获取所有节点的边
        if offset >= len(self.__vertexSet):
            print("Class GraphSet curVESet() offset out of index!")
            exit()
        vertexNum = len(self.__vertexSet[offset])
        result = [[] for _ in range(vertexNum)]
        for key in self.__edgeSet[offset]:
            v1, v2 = key.strip().split(":")
            result[int(v1) - 1].append(key)
            result[int(v2) - 1].append(key)
        return result

    def neighbor(self, offset, vertex_index) -> list:
        if offset >= len(self.__vertexSet):
            print("Class GraphSet neighbor() offset out of index!")
            exit()

        VESet = self.curVESet(offset)
        return VESet[vertex_index]

    def neighbor_with_type(self, offset, vertex_index) -> dict:
        if offset >= len(self.__vertexSet):
            print("Class GraphSet neighbor() offset out of index!")
            exit()
        neighbors = self.curVESet(offset)[vertex_index]
        res = {}
        before_list = []
        after_list = []
        for edge in neighbors:
            v1, v2 = edge.strip().split(":")
            if v1 == str(vertex_index + 1):
                after_list.append(self.curVSet(offset).get(v2))
                # res.append(Node("after", self.curVSet(offset)[str(int(v2) - 1)]))
            elif v2 == str(vertex_index + 1):
                before_list.append(self.curVSet(offset).get(v1))
                # res.append(Node("before", self.curVSet(offset)[str(int(v1) - 1)]))
        res.setdefault("before", before_list)
        res.setdefault("after", after_list)
        return res


def _extract_graph_from_plan(plan: Plan) -> DiGraph:
    graph = DiGraph()
    for task in plan.tasks:
        task_id, task_type, instruction = task.task_id, task.task_type, task.instruction
        graph.add_node(task_id, task_type=task_type, instruction=instruction)
        for dep_id in task.dependent_task_ids:
            graph.add_edge(dep_id, task_id)
    # # 打印图的节点及其属性
    # print("Nodes and their attributes:")
    # for node, attrs in graph.nodes(data=True):
    #     print(f"Task ID: {node}, Attributes: {attrs}")
    #
    # # 打印图的边
    # print("\nEdges:")
    # for edge in graph.edges():
    #     print(f"Dependency: {edge[0]} -> {edge[1]}")
    return graph


# if __name__ == '__main__':
#     task_list = [Task(task_id='1', dependent_task_ids=[], instruction='List all files in the input directory to understand the available datasets', task_type='pda', code='', result='', is_success=False, is_finished=False), Task(task_id='2', dependent_task_ids=['1'], instruction='Load the training datasets for prompts and summaries', task_type='data preprocessing', code='', result='', is_success=False, is_finished=False), Task(task_id='3', dependent_task_ids=['2'], instruction="Merge the training datasets on 'prompt_id'", task_type='data preprocessing', code='', result='', is_success=False, is_finished=False), Task(task_id='4', dependent_task_ids=['3'], instruction='Select relevant columns for the training dataset and create a combined text column', task_type='feature engineering', code='', result='', is_success=False, is_finished=False), Task(task_id='5', dependent_task_ids=['4'], instruction="Split the training data into features and target variables for 'content' and 'wording' scores", task_type='data preprocessing', code='', result='', is_success=False, is_finished=False), Task(task_id='6', dependent_task_ids=['5'], instruction='Split the data into training and validation sets', task_type='data preprocessing', code='', result='', is_success=False, is_finished=False), Task(task_id='7', dependent_task_ids=['6'], instruction='Vectorize the text data using TfidfVectorizer', task_type='feature engineering', code='', result='', is_success=False, is_finished=False), Task(task_id='8', dependent_task_ids=['7'], instruction="Train a Linear Regression model for the 'content' score", task_type='machine learning', code='', result='', is_success=False, is_finished=False), Task(task_id='9', dependent_task_ids=['7'], instruction="Train a Linear Regression model for the 'wording' score", task_type='machine learning', code='', result='', is_success=False, is_finished=False), Task(task_id='10', dependent_task_ids=['8', '9'], instruction='Evaluate the models using Mean Squared Error', task_type='machine learning', code='', result='', is_success=False, is_finished=False), Task(task_id='11', dependent_task_ids=['1'], instruction='Load the test datasets for prompts and summaries', task_type='data preprocessing', code='', result='', is_success=False, is_finished=False), Task(task_id='12', dependent_task_ids=['11'], instruction="Merge the test datasets on 'prompt_id'", task_type='data preprocessing', code='', result='', is_success=False, is_finished=False), Task(task_id='13', dependent_task_ids=['12'], instruction='Select relevant columns for the test dataset and create a combined text column', task_type='feature engineering', code='', result='', is_success=False, is_finished=False), Task(task_id='14', dependent_task_ids=['13', '7'], instruction='Vectorize the test data using the trained TfidfVectorizer', task_type='feature engineering', code='', result='', is_success=False, is_finished=False), Task(task_id='15', dependent_task_ids=['14', '8', '9'], instruction="Predict the 'content' and 'wording' scores for the test dataset using the trained models", task_type='machine learning', code='', result='', is_success=False, is_finished=False), Task(task_id='16', dependent_task_ids=['15'], instruction='Output the result with print() function.', task_type='other', code='', result='', is_success=False, is_finished=False)]
#     plan = Plan(goal="", tasks=task_list)
#     graph = _extract_graph_from_plan(plan)
