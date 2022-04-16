import numpy as np
import pickle
import torch as t
import itertools as itl


class GraphData:
    def __init__(self, args):
        # self.path = args.path
        self.path = '../data/' + args.dataset + '/'
        self.args = args
        self.train_file = self.path + 'data.train'
        self.eval_file = self.path + 'data.eval'
        self.test_file = self.path + 'data.test'
        para_file = self.path + 'data.para'
        self.user_graphs_file = self.path + 'data.user_graphs'
        self.item_graphs_file = self.path + 'data.item_graphs'
        self.para_data = self.load_data(para_file)
        self.global_graphs_file = self.path + 'myprocess/graph'
        self.statistic_ratings()
        print('number of users:', self.n_users)
        print('number of items:', self.n_items)
        print('number of words:', self.word_num)

    def load_data(self, file):
        file_path = open(file, 'rb')
        data = pickle.load(file_path)
        file_path.close()
        return data

    def statistic_ratings(self):
        self.n_users = self.para_data['user_num']
        self.n_items = self.para_data['item_num']
        self.word2id = self.para_data['vocab']
        self.word_num = len(self.word2id)
        self.train_length = self.para_data['train_length']
        self.eval_length = self.para_data['eval_length']
        self.test_length = self.para_data['test_length']

    def generate_review(self):
        user_review = self.load_data(self.user_graphs_file)
        item_review = self.load_data(self.item_graphs_file)
        return user_review, item_review

    def generate_batch(self, args, type='train'):
        users_id, items_id, rate = [], [], []
        if type == 'train':
            trains = self.load_data(self.train_file)
        elif type == 'eval':
            trains = self.load_data(self.eval_file)
        elif type == 'test':
            trains = self.load_data(self.test_file)
        else:
            raise ValueError('type must be train, eval or test')
        iter_num = len(trains) // args.batch_size
        for batch_num in range(iter_num):
            start_index = batch_num * args.batch_size
            end_index = min((batch_num + 1) * args.batch_size, len(trains))
            for train in trains[start_index:end_index]:
                users_id.append(train[0])
                items_id.append(train[1])
                rate.append(train[2])
            yield users_id, items_id, rate

    def generate_graph(self):
        graph = self.load_data(self.global_graphs_file)
        uedg_index, iedg_index, user, item = graph[0], graph[1], graph[2], graph[3]
        return uedg_index, iedg_index, user, item
