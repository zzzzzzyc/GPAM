import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch
from functools import partial
import torch
from torch.nn import Linear, BatchNorm1d, Parameter
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
import random
import copy
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from tqdm import tqdm
from torch_geometric.nn.conv import GATConv
from .layers import SAGEConv
from .configuration_graph_transformer import GraphEncoderConfig

class GraphEncoder(nn.Module):
    def __init__(self, config: GraphEncoderConfig):
        super(GraphEncoder, self).__init__()
        self.num_features = config.num_features
        self.num_token = config.num_token
        self.num_out = config.gnn_out
        self.n_layers = config.n_layers
        self.hidden = config.gnn_hidden
        self.num_proj_hidden = config.proj_hidden
        self.activation = F.relu
        self.drop_out = config.drop_out

        self.GT = GraphSAGE(self.num_features,
                            self.hidden,
                            self.num_out,
                            self.n_layers,
                            self.num_proj_hidden,
                            self.activation,
                            self.drop_out
                            )


        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, graph):

        node_embedding = self.GT.forward_graphcl(graph)

        return node_embedding
# class GraphEncoder(nn.Module):
#     def __init__(self, num_features, num_token, gnn_hidden, gnn_out, proj_hidden):
#         super(GraphEncoder, self).__init__()
#         self.num_features = num_features
#         self.num_token = num_token
#         self.hidden = gnn_hidden
#         self.gnn_output = gnn_out
#         self.num_proj_hidden = proj_hidden
#
#         # 假设 BiGCN_graphcl 是你已有的 GNN 模块
#         self.GT = BiGCN_graphcl(self.num_features, self.hidden, self.gnn_output, self.num_proj_hidden)
#
#         self.graph_projector = nn.Sequential(
#             nn.Linear(self.gnn_output, self.num_token * self.gnn_output)
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.LayerNorm):
#                 m.weight.data.fill_(1.0)
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(mean=0.0, std=1.0)
#
#     def forward(self, graph):
#         node_embedding = self.graph_projector(self.GT.forward_graphcl(graph))  # [B, num_token * gnn_output]
#         return node_embedding
class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_norm (bool, optional): whether or not to normalize adj matrix.
            (default: :obj:`True`)
        gfn (bool, optional): If `True`, only linear transform (1x1 conv) is
            applied to every nodes. (default: :obj:`False`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 edge_norm=True,
                 gfn=False):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes,),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        edge_index = edge_index[0]
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(
                    edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, tddroprate):
        super(TDrumorGCN, self).__init__()
        self.tddroprate = tddroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x, edge_index = data.x, data.edge_index

        edge_index_list = edge_index.tolist()
        if self.tddroprate > 0:
            length = len(edge_index_list[0])
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            tdrow = list(np.array(edge_index_list[0])[poslist])
            tdcol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([tdrow, tdcol]).to(device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, budroprate):
        super(BUrumorGCN, self).__init__()
        self.budroprate = budroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x = data.x
        edge_index = data.edge_index.clone()
        edge_index[0], edge_index[1] = data.edge_index[1], data.edge_index[0]

        edge_index_list = edge_index.tolist()
        if self.budroprate > 0:
            length = len(edge_index_list[0])
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            burow = list(np.array(edge_index_list[0])[poslist])
            bucol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([burow, bucol]).to(device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x

class BiGCN_graphcl(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_proj_hidden,tddroprate=0.0, budroprate=0.0):
        super(BiGCN_graphcl, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, tddroprate)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, budroprate)
        self.proj_head = torch.nn.Linear((out_feats + hid_feats) * 2, out_feats)

        # non-linear layer for contrastive loss
        self.fc1 = torch.nn.Linear(out_feats, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, out_feats)

    # def forward(self, data):
    #     TD_x = self.TDrumorGCN(data)
    #     BU_x = self.BUrumorGCN(data)
    #     x = torch.cat((BU_x, TD_x), 1)
    #     x = self.fc(x)
    #     return F.log_softmax(x, dim=-1)

    def forward_graphcl(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.proj_head(x)
        return x

    def loss_graphcl(self, x1, x2, mean=True):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, num_proj_hidden, activation, dropout,
                 graph_pooling='sum', edge_dim=None):
        super().__init__()

        self.n_layers = n_layers
        self.n_hidden = hidden_channels
        self.n_classes = out_channels
        self.convs = torch.nn.ModuleList()
        gnn_conv = SAGEConv

        if n_layers > 1:
            self.convs.append(gnn_conv(in_channels, hidden_channels))
            for i in range(1, n_layers - 1):
                self.convs.append(gnn_conv(hidden_channels, hidden_channels))
            self.convs.append(gnn_conv(hidden_channels, out_channels))
        else:
            self.convs.append(gnn_conv(in_channels, out_channels))

        # non-linear layer for contrastive loss
        self.fc1 = torch.nn.Linear(out_channels, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, out_channels)


        self.dropout = dropout
        self.activation = activation

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool

    def forward_graphcl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = x.to(self.convs[0].lin_l.weight.dtype)

        for i, conv in enumerate(self.convs):
            # x = conv(x, edge_index, edge_attr=edge_attr)
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = list(unbatch(x, batch))
        xs = []
        for i in range(len(x)):
            xs.append(x[i][0])
        x = t.stack(xs, dim=0)

        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

