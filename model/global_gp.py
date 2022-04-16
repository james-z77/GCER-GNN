import sys

import scipy.sparse as sp
import pickle
import networkx as nx
import numpy as np

dataset = sys.argv[1]
threshold = float(sys.argv[2])


def get_edge():
    uedge_list = set()
    iedge_list = set()
    a = open('../data/' + dataset + '/myprocess/userdata_train.csv', 'rb')
    b = open('../data/' + dataset + '/myprocess/itemdata_train.csv', 'rb')
    [user_matrix, user_keys] = pickle.load(a)
    [item_matrix, item_keys] = pickle.load(b)

    for i in range(len(user_matrix[0])):
        uedge_list.add((user_keys[i], user_keys[i]))
        for j in range(i):
            if user_matrix[i][j] >= threshold:
                uedge_list.add((user_keys[i], user_keys[j]))
    for i in range(len(item_matrix[0])):
        iedge_list.add((item_keys[i], item_keys[i]))
        for j in range(i):
            if item_matrix[i][j] >= threshold:
                iedge_list.add((item_keys[i], item_keys[j]))

    return list(uedge_list), list(iedge_list), user_keys, item_keys


def generate_graph():
    uedge_list, iedge_list, user, item = get_edge()
    uG = nx.Graph()
    iG = nx.Graph()
    uG.add_nodes_from(user)
    iG.add_nodes_from(item)
    uG.add_edges_from(uedge_list)
    iG.add_edges_from(iedge_list)
    uadj_matrix = nx.to_scipy_sparse_matrix(uG)
    iadj_matrix = nx.to_scipy_sparse_matrix(iG)
    uindex_val = sp.find(uadj_matrix)
    iindex_val = sp.find(iadj_matrix)
    uedg_index = np.array([uindex_val[1], uindex_val[0]])
    iedg_index = np.array([iindex_val[1], iindex_val[0]])

    return uedg_index, iedg_index, uG.nodes, iG.nodes


uedg_index, iedg_index, user, item = generate_graph()

l = [uedg_index, iedg_index, list(user), list(item)]
print(uedg_index.shape)
print(len(user))
a = open('../data/' + dataset + '/myprocess/graph', 'wb')
pickle.dump(l, a)
