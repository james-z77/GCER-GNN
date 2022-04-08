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


def train_model(args, uedg_index, iedg_index, user, item, user_id, item_id, rate_, user_review, item_review, user_test,
                item_test, rate_test):
    print('################Training model###################')
    stopping_step = 0
    best_cu = None
    for epoch in range(1, args.epochs + 1):
        t1 = time()
        zu = np.zeros((len(user), args.dim))
        zi = np.zeros((len(item), args.dim))
        zu, zi = train(user, item, zu, zi, user_review, item_review)
        mytrain(uedg_index, iedg_index, user, item, zu, zi, user_id, item_id, rate_)

        test_mse = mytest(uedg_index, iedg_index, user, item, zu, zi, user_test, item_test, rate_test)
        t2 = time()
        print('epoch{: d}: train_time:{: .2f}s'.format(
            epoch, t2 - t1))
        best_cu, stopping_step, should_stop = early_stopping(
            test_mse, best_cu, test_mse, stopping_step, flag_step=300)
        if should_stop:
            break
    print('best mse: {:.4f}'.format(test_value_r))


def train(user, item, zu, zi, user_review, item_review):
    iter_num = int(len(user) / args.batch_size) + 1
    l = list(range(len(user)))
    np.random.shuffle(l)
    for batch_num in range(iter_num):
        start_index = batch_num * args.batch_size
        end_index = min((batch_num + 1) * args.batch_size, len(user))
        u_sub = []
        uid_batch = []

        ll = l[start_index:end_index]
        for i in ll:
            uid_batch.append(i)
            adj_ind, adj_val, node_index, sub = user_review[i]
            u_sub.append(sub)
        uid_batch = gpu(uid_batch)
        sub_u = []
        for u in u_sub:
            sub_uu = []
            for uu in u:
                sub_uu.append([gpu(it) for it in uu])
            sub_u.append(sub_uu)

        user_vc = model(uid_batch, sub_u)
        for i, id in enumerate(uid_batch):
            it = int(id.cpu())
            zu[it][:] = user_vc[i].cpu().detach().numpy()

    iter_num = int(len(item) / args.batch_size) + 1
    l = list(range(len(item)))
    np.random.shuffle(l)
    for batch_num in range(iter_num):
        start_index = batch_num * args.batch_size
        end_index = min((batch_num + 1) * args.batch_size, len(item))
        i_sub = []
        iid_batch = []
        ll=l[start_index:end_index]
        for i in ll:
            iid_batch.append(i)
            adj_ind, adj_val, node_index, sub = item_review[i]
            i_sub.append(sub)

        iid_batch = gpu(iid_batch)

        sub_i = []
        for i in i_sub:
            sub_ii = []
            for ii in i:
                sub_ii.append([gpu(it) for it in ii])
            sub_i.append(sub_ii)
        item_vc = model(iid_batch, sub_i)
        for i, id in enumerate(iid_batch):
            it = int(id.cpu())
            zi[it][:] = item_vc[i].cpu().detach().numpy()

    return zu, zi


def mytrain(uedg_index, iedg_index, user, item, zu, zi, user_id, item_id, rate_):
    mymodel.train()
    optimizer1.zero_grad()
    model.train()
    optimizer2.zero_grad()
    uout = t.index_select(t.from_numpy(zu), dim=0, index=user.cpu())
    iout = t.index_select(t.from_numpy(zi), dim=0, index=item.cpu())
    pre = mymodel(uedg_index, iedg_index, user, item, uout, iout, user_id, item_id)
    loss = loss_function(pre, t.tensor(rate_).cuda())
    print(loss.item() / len(rate_))
    loss.backward()
    optimizer1.step()
    optimizer2.step()


def mytest(uedg_index, iedg_index, user, item, zu, zi, user_id, item_id, rate_):
    mymodel.eval()
    optimizer1.zero_grad()
    model.eval()
    optimizer2.zero_grad()
    uout = t.index_select(t.from_numpy(zu), dim=0, index=user.cpu())
    iout = t.index_select(t.from_numpy(zi), dim=0, index=item.cpu())
    pre = mymodel(uedg_index, iedg_index, user, item, uout, iout, user_id, item_id)
    loss = loss_function(pre, t.tensor(rate_).cuda())
    print("test")
    print(loss.item() / len(rate_))
    return loss.item() / len(rate_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='music', help='type of dataset')
    parser.add_argument('--batch_size', type=int, default=1280,
                        help='size of batch of data')
    parser.add_argument('--num_layers', type=int,
                        default=2, help='number of GAT layer')
    parser.add_argument('--dim', type=int, default=8,
                        help='dim of user/item embedding')
    parser.add_argument('--word_dim', type=int, default=16,
                        help='dim of word embedding')
    parser.add_argument('--hidd_dim', type=int, default=8,
                        help='dim of graph node in TGAT')
    parser.add_argument('--factors', type=int, default=16,
                        help='dim of bilinear interaction')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--l2_re', type=float,
                        default=1.0, help='weight of l2')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epoch')
    parser.add_argument('--dropout', type=float,
                        default=0.7, help='dropout rate')
    args = parser.parse_known_args()[0]
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
    a = open('../data/' + args.dataset + '/myprocess/graph', 'rb')
    b = open('../data/' + args.dataset + '/data.train', 'rb')
    c = open('../data/' + args.dataset + '/data.user_graphs', 'rb')
    d = open('../data/' + args.dataset + '/data.item_graphs', 'rb')
    e = open('../data/' + args.dataset + '/data.test', 'rb')
    rates = pickle.load(b)
    rates_test = pickle.load(e)
    user_review = pickle.load(c)
    item_review = pickle.load(d)
    user_id, item_id, rate_ = [], [], []
    for rate in rates:
        user_id.append(rate[0])
        item_id.append(rate[1])
        rate_.append(rate[2])
    user_test, item_test, rate_test = [], [], []
    for rate in rates_test:
        user_test.append(rate[0])
        item_test.append(rate[1])
        rate_test.append(rate[2])
    graph = pickle.load(a)
    uedg_index, iedg_index, user, item = graph[0], graph[1], graph[2], graph[3]

    uedg_index = np.array(uedg_index)
    uedg_index = gpu(uedg_index)
    iedg_index = np.array(iedg_index)
    print(uedg_index.shape)
    print(iedg_index.shape)
    iedg_index = gpu(iedg_index)
    user = gpu([int(x) for x in user])
    item = gpu([int(x) for x in item])
    try:
        train_model(args, uedg_index, iedg_index, user, item, user_id, item_id, rate_, user_review, item_review,
                    user_test, item_test, rate_test)
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
