import os
import pickle

import numpy as np
import sys


def process():
    dataset = sys.argv[1]
    path = '../data/' + dataset + '/pro_data/'
    _type = ['data_train.csv', 'data_valid.csv', 'data_test.csv']
    for x in _type:
        f = open(os.path.join(path + x))
        dist_user = {}
        dist_item = {}
        for lines in f:
            line = lines.split(',')
            if line[0] in dist_user.keys():
                dist_user[line[0]].add(line[1])
            else:
                set_user = set(line[1])
                dist_user[line[0]] = set_user
            if line[1] in dist_item.keys():
                dist_item[line[1]].add(line[0])
            else:
                set_item = set(line[0])
                dist_item[line[1]] = set_item
        user_keys = list(dist_user.keys())
        user_number = len(user_keys)
        user_matrix = np.zeros([user_number, user_number])
        for i in range(user_number):
            for j in range(i):
                if i == j or len(dist_user[user_keys[i]].union(dist_user[user_keys[j]])) == 0:
                    continue
                else:
                    union_size = len(dist_user[user_keys[i]].union(dist_user[user_keys[j]]))
                    inter_size = len(dist_user[user_keys[i]].intersection(dist_user[user_keys[j]]))
                    jaccard = inter_size / union_size
                    user_matrix[i][j] = jaccard
                    user_matrix[j][i] = jaccard
        item_keys = list(dist_item.keys())
        item_number = len(item_keys)
        item_matrix = np.zeros([item_number, item_number])
        for i in range(item_number):
            for j in range(i):
                if i == j or len(dist_item[item_keys[i]].union(dist_item[item_keys[j]])) == 0:
                    continue
                else:
                    union_size = len(dist_item[item_keys[i]].union(dist_item[item_keys[j]]))
                    inter_size = len(dist_item[item_keys[i]].intersection(dist_item[item_keys[j]]))
                    jaccard = inter_size / union_size
                    item_matrix[i][j] = jaccard
                    item_matrix[j][i] = jaccard
        a = open(os.path.join('../data/' + dataset + '/myprocess/user' + x), 'wb')
        b = open(os.path.join('../data/' + dataset + '/myprocess/item' + x), 'wb')

        pickle.dump([user_matrix, user_keys], a)
        pickle.dump([item_matrix, item_keys], b)
        print(11)


if __name__ == '__main__':
    process()
