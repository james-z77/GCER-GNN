import pickle
import torch as t
import torch.optim as optim
from data_loader import GraphData
from model import RGNN
from model import mygat
import numpy as np
import argparse
import random
from time import time


def gpu(batch, is_long=True, use_cuda=True):
    if is_long:
        batch = t.LongTensor(batch)
    else:
        batch = t.FloatTensor(batch)
    if use_cuda:
        batch = batch.cuda()
    return batch


def early_stopping(log_value, best_value, test_mse, stopping_step, flag_step=3):
    global test_value_r
    global test_value_m
    if best_value is None or log_value <= best_value:
        stopping_step = 0
        best_value = log_value
        test_value_r = test_mse
    else:
        stopping_step += 1
    if stopping_step >= flag_step:
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def train_model(data_generator):
    print('################Training model###################')
    f.write('################Training model###################\n')
    stopping_step = 0
    best_cu = None
    uedg_index, iedg_index, user, item = data_generator.generate_graph()
    user_matrix = t.rand((data_generator.n_users, args.dim))
    item_matrix = t.rand((data_generator.n_items, args.dim))
    user = [int(i) for i in user]
    item = [int(i) for i in item]
    uedg_index = np.array(uedg_index)
    uedg_index = gpu(uedg_index)
    iedg_index = np.array(iedg_index)
    iedg_index = gpu(iedg_index)
    user_review, item_review = data_generator.generate_review()
    for epoch in range(1, args.epochs + 1):
        t1 = time()
        train_mse = train(data_generator, user_matrix, item_matrix, user_review, item_review, uedg_index, iedg_index,
                          user, item)
        print('Epoch: ', epoch, '\tTime: ', time() - t1)
        f.write('Epoch: ' + str(epoch) + '\tTime: ' + str(time() - t1) + '\n')
        print("Train Mse Is: ", train_mse)
        test_mse = test(data_generator, user_matrix, item_matrix, user_review, item_review, uedg_index, iedg_index,
                        user, item)
        print("Test Mse Is: ", test_mse)
        f.write("Test Mse Is: " + str(test_mse) + '\n')
        best_cu, stopping_step, should_stop = early_stopping(
            test_mse, best_cu, test_mse, stopping_step, flag_step=3)
        if should_stop:
            break
    print('best mse: {:.4f}'.format(test_value_r))
    f.write('best mse: {:.4f}'.format(test_value_r) + '\n')
    f.close()


def find_index(a, b):
    x = []
    for i in a:
        x.append(b.index(i))
    return x


def train(data_generator, user_matrix, item_matrix, user_review, item_review, uedg_index, iedg_index, user, item):
    mymodel.train()
    model.train()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    epoch_mse = 0.0

    for users_id, items_id, rates in data_generator.generate_batch(args):
        users_id = users_id[-args.batch_size:]
        items_id = items_id[-args.batch_size:]
        rates = rates[-args.batch_size:]

        rates = gpu(rates)
        u_sub_list, i_sub_list = [], []

        for user_id in users_id:
            _, _, _, u_sub = user_review[user_id]
            for every_single_review in u_sub:
                for i in range(len(every_single_review)):
                    if not t.is_tensor(every_single_review[i]):
                        every_single_review[i] = gpu(every_single_review[i])
            u_sub_list.append(u_sub)

        for item_id in items_id:
            _, _, _, i_sub = item_review[item_id]
            for every_single_review in i_sub:
                for i in range(len(every_single_review)):
                    if not t.is_tensor(every_single_review[i]):
                        every_single_review[i] = gpu(every_single_review[i])
            i_sub_list.append(i_sub)

        users_id = gpu(users_id)
        items_id = gpu(items_id)

        user_vc = model(users_id, u_sub_list)
        item_vc = model(items_id, i_sub_list)

        user_index = find_index(users_id, user)
        item_index = find_index(items_id, item)

        for i, index in enumerate(user_index):
            user_matrix[index] = user_vc[i].cpu().detach()
        for i, index in enumerate(item_index):
            item_matrix[index] = item_vc[i].cpu().detach()

        pre = mymodel(uedg_index, iedg_index, user_matrix, item_matrix, users_id, items_id, user, item)
        del user_vc, item_vc, user_index, item_index, users_id, items_id, u_sub_list, i_sub_list
        loss = loss_function(pre, rates.to(t.float32))
        epoch_mse += loss.item()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
    train_mse = epoch_mse / data_generator.train_length
    return train_mse


def test(data_generator, user_matrix, item_matrix, user_review, item_review, uedg_index, iedg_index, user, item):
    mymodel.eval()
    model.eval()
    epoch_mse = 0.0

    for users_id, items_id, rates in data_generator.generate_batch(args, type='test'):
        users_id = users_id[-args.batch_size:]
        items_id = items_id[-args.batch_size:]
        rates = rates[-args.batch_size:]
        rates = gpu(rates)
        u_sub_list, i_sub_list = [], []

        for user_id in users_id:
            _, _, _, u_sub = user_review[user_id]
            for every_single_review in u_sub:
                for i in range(len(every_single_review)):
                    if not t.is_tensor(every_single_review[i]):
                        every_single_review[i] = gpu(every_single_review[i])
            u_sub_list.append(u_sub)

        for item_id in items_id:
            _, _, _, i_sub = item_review[item_id]
            for every_single_review in i_sub:
                for i in range(len(every_single_review)):
                    if not t.is_tensor(every_single_review[i]):
                        every_single_review[i] = gpu(every_single_review[i])
            i_sub_list.append(i_sub)

        users_id = gpu(users_id)
        items_id = gpu(items_id)

        user_vc = model(users_id, u_sub_list)
        item_vc = model(items_id, i_sub_list)

        user_index = find_index(users_id, user)
        item_index = find_index(items_id, item)

        for i, index in enumerate(user_index):
            user_matrix[index] = user_vc[i].cpu().detach()
        for i, index in enumerate(item_index):
            item_matrix[index] = item_vc[i].cpu().detach()

        pre = mymodel(uedg_index, iedg_index, user_matrix, item_matrix, users_id, items_id, user, item)
        del user_vc, item_vc, user_index, item_index, users_id, items_id, u_sub_list, i_sub_list
        with t.no_grad():
            loss = loss_function(pre, rates.to(t.float32))
            epoch_mse += loss.item()
    train_mse = epoch_mse / data_generator.test_length
    return train_mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='music', help='type of dataset')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of batch of data')
    parser.add_argument('--num_layers', type=int,
                        default=2, help='number of GAT layer')
    parser.add_argument('--dim', type=int, default=32,
                        help='dim of user/item embedding')
    parser.add_argument('--word_dim', type=int, default=32,
                        help='dim of word embedding')
    parser.add_argument('--hidd_dim', type=int, default=32,
                        help='dim of graph node in TGAT')
    parser.add_argument('--factors', type=int, default=10,
                        help='dim of bilinear interaction')

    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--l2_re', type=float,
                        default=0.1, help='weight of l2')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epoch')
    parser.add_argument('--dropout', type=float,
                        default=0.2, help='dropout rate')
    args = parser.parse_known_args()[0]
    f = open("result" + args.dataset + ".txt", 'w')
    f.write('dataset: ' + args.dataset + '\n')
    f.write('batch_size: ' + str(args.batch_size) + '\n')
    f.write('num_layers: ' + str(args.num_layers) + '\n')
    f.write('dim: ' + str(args.dim) + '\n')
    f.write('word_dim: ' + str(args.word_dim) + '\n')
    f.write('hidd_dim: ' + str(args.hidd_dim) + '\n')
    f.write('factors: ' + str(args.factors) + '\n')
    f.write('lr: ' + str(args.lr) + '\n')
    f.write('l2_re: ' + str(args.l2_re) + '\n')
    f.write('epochs: ' + str(args.epochs) + '\n')
    f.write('dropout: ' + str(args.dropout) + '\n')
    np.random.seed(1999)
    random.seed(1999)
    t.manual_seed(1999)
    t.cuda.manual_seed_all(1999)
    data_generator = GraphData(args)
    test_value_r = 0
    loss_function = t.nn.MSELoss(reduction='sum')

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_words'] = data_generator.word_num
    model = RGNN(config=config, args=args)
    mymodel = mygat(config=config, args=args)
    model = model.cuda()
    mymodel.cuda()
    optimizer1 = optim.Adam(mymodel.parameters(), lr=args.lr,
                            weight_decay=args.l2_re)
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.l2_re)

    train_model(data_generator)
