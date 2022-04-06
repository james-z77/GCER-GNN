import os
import csv
import numpy as np
import json
import pandas as pd
from nltk.corpus import stopwords
import pickle
import random
import sys
from nltk.stem import PorterStemmer


def process():
    dataset="toys"
    path = '../data/' + dataset + '/pro_data/'
    _type = ['data_train.csv', 'data_valid.csv', 'data_test.csv']
    for x in _type:
        f = open(os.path.join(path + x))
        d_u = {}
        d_i = {}
        for lines in f:
            line = lines.split(',')
            if line[0] in d_u.keys():
                d_u[line[0]].append(line[1])
            else:
                l_u = [line[1]]
                d_u[line[0]] = l_u
            if line[1] in d_i.keys():
                d_i[line[1]].append(line[0])
            else:
                l_i = [line[0]]
                d_i[line[1]] = l_i
        dst_u = {}
        dst_i = {}
        for k in d_u.keys():
            dic = {}
            dst_u[k] = dic
            for i in d_u[k]:
                for u in d_i[i]:
                    if u == k:
                        continue
                    if len(dst_u[k]) == 0:
                        dst_u[k][u] = 1
                        continue
                    if (u in dst_u[k].keys()):
                        dst_u[k][u] += 1
                    else:
                        dst_u[k][u] = 1
        for k in d_i.keys():
            dic = {}
            dst_i[k] = dic
            for i in d_i[k]:
                for u in d_u[i]:
                    if u == k:
                        continue
                    if len(dst_i[k]) == 0:
                        dst_i[k][u] = 1
                        continue
                    if (u in dst_i[k].keys()):
                        dst_i[k][u] += 1
                    else:
                        dst_i[k][u] = 1
        a = open(os.path.join('../data/' + dataset + '/myprocess/user' + x), 'wb')
        b = open(os.path.join('../data/' + dataset + '/myprocess/item' + x), 'wb')
        print(11)
        pickle.dump(dst_u, a)
        pickle.dump(dst_i, b)


if __name__ == '__main__':
    process()
