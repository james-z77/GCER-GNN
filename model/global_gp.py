import sys

import scipy.sparse as sp
import pickle
import networkx as nx
import numpy as np

dataset = sys.argv[1]
threshold = 0.4


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


uedge_list, iedge_list, user, item = get_edge()
l = [[[int(c[0]), int(c[1])] for c in uedge_list].reverse(), [[int(c[0]), int(c[1])] for c in iedge_list].reverse(), list(user), list(item)]
a = open('../data/' + dataset + '/myprocess/graph', 'wb')
pickle.dump(l, a)
