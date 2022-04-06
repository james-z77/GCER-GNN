import scipy.sparse as sp
import pickle
import networkx as nx
import numpy as np
dataset="toys"

def get_edge():
    uedge_list = set()
    iedge_list = set()
    a = open('../data/' + dataset + '/myprocess/userdata_train.csv', 'rb')
    b = open('../data/' + dataset + '/myprocess/itemdata_train.csv', 'rb')
    user = pickle.load(a)
    item = pickle.load(b)
    for k in user.keys():
        uedge_list.add((k,k))
        l = user[k]
        for u in l.keys():
            if l[u] >= 5:
                uedge_list.add((k, u))
    for k in item.keys():
        iedge_list.add((k, k))
        l = item[k]
        for u in l.keys():
            if l[u] >= 5:
                iedge_list.add((k, u))
    return list(uedge_list), list(iedge_list), list(user.keys()), list(item.keys())


def generate_graph():
    uedge_list, iedge_list, user, item = get_edge()
    print(len(uedge_list))
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
a = open('../data/' + dataset + '/myprocess/graph', 'wb')
pickle.dump(l, a)
