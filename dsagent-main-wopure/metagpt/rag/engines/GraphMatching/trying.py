import time
from collections import Counter

from scipy.optimize import linear_sum_assignment

from metagpt.actions.ds_agent.fixed_plan_for_test import get_fixed_plan
from metagpt.rag.engines.GraphMatching import KMMatcher
from metagpt.rag.engines.GraphMatching.graph import GraphSet

plan1 = get_fixed_plan(181)
plan1_2 = get_fixed_plan(1812)
plan2 = get_fixed_plan(57)


def getGraphEditDistance(graph1: GraphSet, graph2: GraphSet):
    n, m = len(graph1.curVSet(0)), len(graph2.curVSet(0))
    matrix = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            dis = 0
            if graph1.curVSet(0).get(str(i + 1)) != graph2.curVSet(0).get(str(j + 1)):
                dis += 1
            g1_neighbors = graph1.neighbor_with_type(0, i)
            g2_neighbors = graph2.neighbor_with_type(0, j)
            g1_before, g1_after = g1_neighbors.get('before', []), g1_neighbors.get('after', [])
            g2_before, g2_after = g2_neighbors.get('before', []), g2_neighbors.get('after', [])

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


def transpose_matrix(matrix):
    return [list(row) for row in zip(*matrix)]


if __name__ == '__main__':
    g1 = GraphSet(plan1)
    g1_2 = GraphSet(plan1_2)
    g2 = GraphSet(plan2)
    print(g1.curVESet(0))
    print(g1_2.curVESet(0))
    print(g2.curVESet(0))
    edit_distance = getGraphEditDistance(g1, g1_2)
    print(f"The graph edit distance is: {edit_distance}")

    # fixme: KMMatcher need to fix
    # row_len, col_len = len(weights), len(weights[0])
    # if row_len > col_len:
    #     weights = transpose_matrix(weights)
    # row_len, col_len = len(weights), len(weights[0])
    # st = time.time()
    # matcher = KMMatcher(weights)
    # edit_distance = matcher.solve()
    # ed = time.time()
    # print('edit distance: ', edit_distance)
    # print('time consuming of size ({}, {}) is {:.4f} seconds'.format(row_len, col_len, ed - st))
