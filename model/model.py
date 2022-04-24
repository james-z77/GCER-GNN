import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.data import Data
import torch_geometric.data as gda
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GATConv
from layer import GraphPool, GAT_LAYER
from layer import GAT
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as back


def gpu(batch, is_long=True, use_cuda=True):
    if t.is_tensor(batch):
        return batch.cuda()
    if is_long:
        batch = t.LongTensor(batch)
    else:
        batch = t.FloatTensor(batch)
    if use_cuda:
        batch = batch.cuda()
    return batch


class RGNN(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.user_embedding = nn.Embedding(config['n_users'], args.dim)
        self.item_embedding = nn.Embedding(config['n_items'], args.dim)
        self.word_embedding = nn.Embedding(config['n_words'], args.word_dim)
        self.agg_u = nn.Linear(
            args.hidd_dim, args.dim)
        self.agg_i = nn.Linear(
            args.hidd_dim, args.dim)
        self.conv_u = nn.ModuleList(
            [GAT(args.word_dim, args.hidd_dim, num_relation=4)])
        self.conv_i = nn.ModuleList(
            [GAT(args.word_dim, args.hidd_dim, num_relation=4)])
        self.GRU_layer = nn.GRU(input_size=args.dim, hidden_size=args.dim, batch_first=True, num_layers=3)
        for _ in range(args.num_layers - 1):
            self.conv_u.append(
                GAT(args.hidd_dim, args.hidd_dim, num_relation=4))
            self.conv_i.append(
                GAT(args.hidd_dim, args.hidd_dim, num_relation=4))
        self.trans_u = nn.ModuleList(
            [nn.Linear(args.dim, args.hidd_dim) for _ in range(args.num_layers)])
        self.trans_i = nn.ModuleList(
            [nn.Linear(args.dim, args.hidd_dim) for _ in range(args.num_layers)])
        self.trans_w = nn.ModuleList(
            [nn.Linear(args.hidd_dim, args.dim) for _ in range(args.num_layers)])
        self.interaction_u = nn.Linear(args.dim, args.dim)
        self.interaction_i = nn.Linear(args.dim, args.dim)
        self.fm2 = FM_Layer(args, config)
        self.Batch = gda.Batch()
        self.Dropout = nn.Dropout(args.dropout)
        self.MYGAT = gat(config, args)

        init.xavier_uniform_(self.user_embedding.weight)
        init.xavier_uniform_(self.item_embedding.weight)
        init.xavier_uniform_(self.word_embedding.weight)

    def step(self, uid_batch, sub_u):
        self.u_e = self.user_embedding(uid_batch)
        usub_temp = []
        uusub_temp = []
        seq_len = []
        for u in sub_u:
            seq_len.append(len(u))
            for i in u:
                ix = self.word_embedding(gpu(i[2]))
                i[0] = gpu(i[0])
                i[1] = gpu(i[1])
                uusub_temp.append(Data(x=ix, edge_index=i[0], edge_attr=i[1].unsqueeze(1)))

        usub_graph = self.Batch.from_data_list(uusub_temp)

        u_pool_e = self.conv_pool(usub_graph, self.conv_u)
        index = 0
        for i in range(len(uid_batch)):
            usub_temp.append(u_pool_e[index:index + seq_len[i]][:])
            index = index + seq_len[i]

        train = pad(usub_temp, batch_first=True)
        seq_len = [s.size(0) for s in usub_temp]
        train = pack(train, seq_len, batch_first=True, enforce_sorted=False)
        x, self.hidden = self.GRU_layer(train)
        x, _ = back(x, batch_first=True)

        user_vc = []
        for i, vc in enumerate(x):
            user_vc.append(vc[seq_len[i] - 1][:].view(self.args.dim))
        user_vc = t.stack(user_vc, dim=0)
        return user_vc

    def forward(self, uid_batch, sub_u, iid_batch, sub_i, uedg_index, iedg_index,uedg_value, iedg_value):
        user_vc1 = self.step(uid_batch, sub_u)
        item_vc1 = self.step(iid_batch, sub_i)
        user_vc2, item_vc2 = self.MYGAT(uedg_index, iedg_index, uid_batch, iid_batch,uedg_value, iedg_value)
        user_vc = t.cat((user_vc1, user_vc2), -1)
        item_vc = t.cat((item_vc1, item_vc2), -1)
        pre_rate = self.fm2(user_vc, item_vc, uid_batch, iid_batch)
        return pre_rate

    def conv_pool(self, graph, conv_ui):
        review_e, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch.cuda()  # graph.batch用于标注每个节点属于第几个图
        review_e = conv_ui[0](review_e, batch, edge_index, edge_attr.squeeze())
        out = gmp(review_e, batch)
        return out


class FM_Layer(nn.Module):
    def __init__(self, args, config):
        super(FM_Layer, self).__init__()
        input_dim = args.dim * 4
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.V = nn.Parameter(
            t.zeros(input_dim, input_dim), requires_grad=True)
        self.bias_u = nn.Parameter(
            t.zeros(config['n_users'], requires_grad=True))
        self.bias_i = nn.Parameter(
            t.zeros(config['n_items'], requires_grad=True))
        self.bias = nn.Parameter(t.zeros(1, requires_grad=True))

        init.xavier_uniform_(self.V.data)

    def fm_layer(self, user_em, item_em, uid, iid):
        x = t.cat((user_em, item_em), -1).unsqueeze(1)
        linear_part = self.linear(x).squeeze()
        batch_size = len(x)
        V = t.stack((self.V,) * batch_size)
        interaction_part_1 = t.bmm(x, V)
        interaction_part_1 = t.pow(interaction_part_1, 2)
        interaction_part_2 = t.bmm(t.pow(x, 2), t.pow(V, 2))
        mlp_output = 0.5 * \
                     t.sum((interaction_part_1 - interaction_part_2).squeeze(1), -1)
        rate = linear_part + mlp_output + \
               self.bias_u[uid] + self.bias_i[iid] + self.bias
        return rate

    def forward(self, user_em, item_em, uid, iid):
        return self.fm_layer(user_em, item_em, uid, iid).view(-1)


class gat(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.umygat1 = GAT_LAYER(in_channels=args.dim,

                                 out_channels=args.dim, dropout=args.dropout)
        self.imygat1 = GAT_LAYER(in_channels=args.dim,
                                 out_channels=args.dim, dropout=args.dropout)
        self.umygat2 = GAT_LAYER(in_channels=args.dim,
                                 out_channels=args.dim, dropout=args.dropout)
        self.imygat2 = GAT_LAYER(in_channels=args.dim,
                                 out_channels=args.dim, dropout=args.dropout)
        self.user_matrix = t.nn.Parameter(gpu(t.rand((config['n_users'], args.dim)), is_long=False), requires_grad=True)

        self.item_matrix = t.nn.Parameter(gpu(t.rand((config['n_items'], args.dim)), is_long=False), requires_grad=True)

    def forward(self, uedg_index, iedg_index, user_id, item_id, uedg_value, iedg_value):
        ugraph = Data(x=self.user_matrix, edge_index=uedg_index)
        igraph = Data(x=self.item_matrix, edge_index=iedg_index)

        iout = self.imygat1(igraph.x, igraph.edge_index, iedg_value)
        uout = self.umygat1(ugraph.x, ugraph.edge_index, uedg_value)
        iout = self.imygat2(iout, igraph.edge_index, iedg_value)
        uout = self.umygat2(uout, ugraph.edge_index, uedg_value)
        user_vc = t.index_select(uout, 0, user_id)
        item_vc = t.index_select(iout, 0, item_id)
        return user_vc, item_vc
