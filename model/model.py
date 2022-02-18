import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.data import Data
import torch_geometric.data as gda
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GATConv
from layer import GraphPool
from layer import GAT
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as back


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
        self.umygat1 = GATConv(in_channels=args.dim, out_channels=args.dim, dropout=0.5)
        self.imygat1 = GATConv(in_channels=args.dim, out_channels=args.dim, dropout=0.5)
        self.umygat2 = GATConv(in_channels=args.dim,
                               out_channels=args.dim, dropout=0.5)
        self.imygat2 = GATConv(in_channels=args.dim,
                               out_channels=args.dim, dropout=0.5)
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
        self.pool = GraphPool(args.hidd_dim)
        self.Batch = gda.Batch()
        self.Dropout = nn.Dropout(args.dropout)

        init.xavier_uniform_(self.user_embedding.weight)
        init.xavier_uniform_(self.item_embedding.weight)
        init.xavier_uniform_(self.word_embedding.weight)

    def find_index(self, a, b):
        c = []
        for i in a:
            mask = (b == i).type(t.ByteTensor)
            c.append(t.nonzero(mask))
        return t.LongTensor(c).cuda()

    def forward(self, uid_batch, sub_u):
        self.u_e = self.user_embedding(uid_batch)
        usub_temp = []
        uusub_temp = []
        seq_len = []
        for u in sub_u:
            seq_len.append(len(u))
            for i in u:
                ix = self.word_embedding(i[2])
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

    def conv_pool(self, graph, conv_ui):
        pool_e = []
        review_e, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch.cuda()  # graph.batch用于标注每个节点属于第几个图
        for i in range(self.args.num_layers):
            review_e = conv_ui[i](
                review_e, batch, edge_index, edge_attr.squeeze())
        review_e, edge_index, edge_attr, batch, _ = self.pool(review_e, edge_index, edge_attr, batch)
        out = gmp(review_e, batch)
        out = t.relu(self.trans_w[i](out))
        pool_e.append(out)
        out_e = t.cat(pool_e, -1)
        return out_e


class FM_Layer(nn.Module):
    def __init__(self, args, config):
        super(FM_Layer, self).__init__()
        input_dim = args.dim * 2
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


class mygat(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.umygat1 = GATConv(in_channels=args.dim,
                               out_channels=args.dim, dropout=args.dropout)
        self.imygat1 = GATConv(in_channels=args.dim,
                               out_channels=args.dim, dropout=args.dropout)
        self.fm2 = FM_Layer(args, config)

    def forward(self, uedg_index, iedg_index, user, item, uout1, iout1, user_id, item_id):
        ugraph = Data(x=uout1.to(t.float32), edge_index=uedg_index.cpu())
        igraph = Data(x=iout1.to(t.float32), edge_index=iedg_index.cpu())
        uout = self.umygat1(ugraph.x, ugraph.edge_index)
        iout = self.imygat1(igraph.x, igraph.edge_index)
        uindex = [int(((user.cpu() == x).nonzero())[0][0]) for x in user_id]
        iindex = [int(((item.cpu() == x).nonzero())[0][0]) for x in item_id]
        uout = t.index_select(uout, dim=0, index=t.tensor(uindex))
        iout = t.index_select(iout, dim=0, index=t.tensor(iindex))
        pre_rate = self.fm2(uout, iout, user_id, item_id)
        return pre_rate
