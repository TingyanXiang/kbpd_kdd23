import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_scatter.composite import scatter_softmax

from model.layers import *

class EdgeV1Model(torch.nn.Module):
    '''
    def massage passing
    '''
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, edge_out_dim
                 ,layer_norm = False, batch_norm = False
                 ,num_layers=2, dropout=0.5
                 ):
        self.global_dim = global_dim
        super(EdgeV1Model, self).__init__()
        self.edgemodel = EdgeModel(node_dim, edge_dim, global_dim, hidden_dim, edge_out_dim
                 ,layer_norm = layer_norm, batch_norm = batch_norm
                 ,num_layers=num_layers, dropout=dropout)
        if self.global_dim is None:
            self.weight_mlp = MLP(edge_dim, hidden_dim, 1, num_layers=1, dropout=dropout, batch_norm=batch_norm, layer_norm=layer_norm)
        else:
            self.weight_mlp = MLP(edge_dim+global_dim, hidden_dim, 1, 1, dropout, batch_norm=batch_norm, layer_norm=layer_norm)

    def forward(self, src, dest, edge_attr, u=None, edge_batch=None):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = self.edgemodel(src, dest, edge_attr, u, edge_batch)

        # 生成权重
        if self.global_dim is None:
            out_1 = edge_attr  # torch.cat([edge_attr], 1)
        else:
            out_1 = torch.cat([edge_attr, u[edge_batch]], 1)
        wts = self.weight_mlp(out_1)  # wts: [#edges, 1]
        unnormalized_wts = wts
        wts = scatter_softmax(wts.squeeze(1), edge_batch, dim=0) #分组qa-pair 进行softmax
        normalized_wts = wts.unsqueeze(1)

        return out, [normalized_wts, unnormalized_wts]

class NodeV1Model(torch.nn.Module):
    '''
    def aggregating and updating
    '''
    def __init__(self, node_dim, edge_dim, global_dim
                 , layer_norm=False, batch_norm=False
                 , message_mlp_hidden_dim=128, message_mlp_num_layers=2, message_mlp_dropout=0.5
                 , update_mlp_hidden_dim=128, update_mlp_num_layers=2, update_mlp_dropout=0.5
                 ):
        super(NodeV1Model, self).__init__()
        # message func
        self.message_mlp = MLP(node_dim+edge_dim, message_mlp_hidden_dim, node_dim,
                               num_layers=message_mlp_num_layers, dropout=message_mlp_dropout, batch_norm=batch_norm, layer_norm=layer_norm)
        # update func
        self.update_mlp = MLP(node_dim*2+global_dim, update_mlp_hidden_dim, node_dim,
                            num_layers=update_mlp_num_layers, dropout=update_mlp_dropout, batch_norm=batch_norm, layer_norm=layer_norm)

    def forward(self, x, edge_index, edge_attr, u, node_batch, wts):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index #row is src
        edge_message = torch.cat([x[row], edge_attr], dim=1)
        edge_message = self.message_mlp(edge_message)  # edge_message: [#edges, hidden_dim]
        if wts is None:
            received_message = scatter_mean(edge_message, col, dim=0, dim_size=x.size(0))
        else:
            received_message = scatter_mean(edge_message * wts, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, received_message, u[node_batch]], dim=1)
        return self.update_mlp(out)



class GraphNetwork(torch.nn.Module):
    '''
        GNN net
    '''
    def __init__(self, edge_model, node_model, num_layers, ablation):
        super(GraphNetwork, self).__init__()
        self.ablation = ablation
        self.edge_model = edge_model
        self.node_model = node_model
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_attr, u, node_batch):
        row, col = edge_index
        edge_batch = node_batch[row]

        for idx in range(self.num_layers):
            # edge attr
            edge_attr, edge_att_outputlist = self.edge_model(x[row], x[col], edge_attr, u, edge_batch)
            normalized_wts, unnormalized_wts = edge_att_outputlist
            unnormalized_wts = torch.sigmoid(unnormalized_wts)
            # node attr
            if 'no_edge_weight' in self.ablation:
                x = self.node_model(x, edge_index, edge_attr, u, node_batch, None)
            else:
                if 'unnormalized_edge_weight' in self.ablation:
                    x = self.node_model(x, edge_index, edge_attr, u, node_batch, unnormalized_wts)
                else:
                    x = self.node_model(x, edge_index, edge_attr, u, node_batch, normalized_wts)
        return x, edge_attr, unnormalized_wts, normalized_wts


class GrailNet(torch.nn.Module):
    def __init__(self, gnn_num_layers=3, nodeatt_heads_num=2
                 , ablation=[]
                 , node_feat_num=None
                 , node_dim=128, edge_dim=128, global_dim=128, hidden_dim=128
                 , mlp_num_layers=2, layer_norm=False, batch_norm=False, dropout=0.5
                 ):
        super(GrailNet, self).__init__()
        self.node_dim = node_dim
        self.gnn_num_layers = gnn_num_layers
        self.ablation = ablation

        edge_model = EdgeV1Model(node_dim, edge_dim, global_dim, hidden_dim, edge_dim
                                 , layer_norm=layer_norm, batch_norm=batch_norm
                                 , num_layers=mlp_num_layers, dropout=dropout
                                 )
        node_model = NodeV1Model(node_dim, edge_dim, global_dim
                                 , layer_norm=layer_norm, batch_norm=batch_norm
                                 , message_mlp_hidden_dim=hidden_dim, message_mlp_num_layers=mlp_num_layers, message_mlp_dropout=dropout
                                 , update_mlp_hidden_dim=hidden_dim, update_mlp_num_layers=mlp_num_layers, update_mlp_dropout=dropout
                                 )
        self.graph_network = GraphNetwork(edge_model, node_model, gnn_num_layers, ablation)
        self.attention = MultiheadAttPoolLayer(nodeatt_heads_num, node_feat_num, hidden_dim)
        # self.attention = MultiheadAttPoolLayer(nodeatt_heads_num, global_dim, hidden_dim)
        self.activation = GELU()
        self.dropout_m = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, u, node_batch, num_of_nodes, qa_attr=None):
        batch_size = u.size(0)
        edge_batch = node_batch[edge_index[0]]
        max_node_num = torch.max(num_of_nodes).item()

        node_vecs, edge_vecs, unnormalized_wts, normalized_wts = self.graph_network(x, edge_index, edge_attr, u, node_batch)
        pooled_edge_vecs = scatter_sum(edge_vecs * normalized_wts, edge_batch, dim=0, dim_size=batch_size)

        evidence_vecs = torch.zeros(batch_size, max_node_num, self.node_dim, device=node_vecs.device)  # 这个device注意一下
        j = 0
        for i in range(batch_size):
            visible_num_tuples = min(num_of_nodes[i].item(), max_node_num)  # 当前graph的节点数量
            evidence_vecs[i, : visible_num_tuples, :] = node_vecs[j: j + visible_num_tuples, :]  # 将graph对应的node vecs 放到evidence_vecs里面
            j = j + num_of_nodes[i].item()
        evidence_vecs = self.activation(evidence_vecs)
        # pooled_node_vecs
        mask = torch.arange(max_node_num, device=node_vecs.device) >= num_of_nodes.unsqueeze(1)
        mask[mask.all(1), 0] = 0

        pooled_node_vecs, att_scores = self.attention(qa_attr, evidence_vecs, mask)
        embeddings = torch.cat((pooled_edge_vecs, pooled_node_vecs), 1)
        return embeddings, [unnormalized_wts, normalized_wts]

class GrailModel(torch.nn.Module):
    def __init__(self, gnn_num_layers=3, nodeatt_heads_num=2, num_class=2
                 , ablation=[]
                 , worknode_feat_num=None, workrelation_feat_num=None, node_feat_num=None
                 , node_dim=128, edge_dim=128, global_dim=128, hidden_dim=128
                 , mlp_num_layers=2, layer_norm=False, batch_norm=False, dropout=0.5
                 ):
        super(GrailModel, self).__init__()
        self.node_dim = node_dim
        self.gnn_num_layers = gnn_num_layers
        # node transform
        self.worknode_transform = nn.Linear(worknode_feat_num, node_dim)
        # edge transform
        self.workrelation_transform = nn.Linear(workrelation_feat_num, edge_dim)
        # global transform
        self.global_transform = MLP(node_feat_num, hidden_dim, global_dim
                                    , num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm,
                                    layer_norm=layer_norm)

        self.ablation = ablation
        self.gnn = GrailNet(gnn_num_layers=gnn_num_layers, nodeatt_heads_num=nodeatt_heads_num
                            , ablation=self.ablation
                            , node_feat_num=node_feat_num
                            , node_dim=node_dim, edge_dim=edge_dim, global_dim=global_dim, hidden_dim=hidden_dim
                            , mlp_num_layers=mlp_num_layers, layer_norm=layer_norm, batch_norm=batch_norm, dropout=dropout
                            )
        self.dropout_m = nn.Dropout(dropout)
        self.hid2out = MLP(node_dim+edge_dim+node_feat_num, hidden_dim, num_class, mlp_num_layers, dropout, batch_norm=batch_norm, layer_norm=batch_norm)

    def forward(self, x, edge_index, edge_attr, qa_attr, node_batch, num_of_nodes):
        # worknode transform
        x = self.worknode_transform(x)
        # workrelation transform
        edge_attr = self.workrelation_transform(edge_attr)
        # global_transform
        u = self.global_transform(qa_attr)
        embeddings, attach_result_list = self.gnn(x, edge_index, edge_attr, u, node_batch, num_of_nodes, qa_attr)
        embeddings = torch.cat((embeddings, qa_attr), 1)
        logits = self.hid2out(self.dropout_m(embeddings))
        attach_result_list = attach_result_list + [embeddings,]
        return logits, attach_result_list

class GrailV1Model(torch.nn.Module):
    def __init__(self, gnn_num_layers=3, nodeatt_heads_num=2, num_class=2
                 , ablation=[]
                 , worknode_feat_num=None, workrelation_feat_num=None, node_feat_num=None
                 , node_dim=128, edge_dim=128, global_dim=128, hidden_dim=128
                 , mlp_num_layers=2, layer_norm=False, batch_norm=False, dropout=0.5
                 ):
        super(GrailV1Model, self).__init__()
        self.node_dim = node_dim
        self.gnn_num_layers = gnn_num_layers
        # node transform
        self.worknode_transform = nn.Linear(worknode_feat_num, node_dim)
        # edge transform
        self.workrelation_transform = nn.Linear(workrelation_feat_num, edge_dim)
        # global transform
        self.global_transform = MLP(node_feat_num, hidden_dim, global_dim
                                    , num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm,
                                    layer_norm=layer_norm)

        edge_model = EdgeV1Model(node_dim, edge_dim, global_dim, hidden_dim, edge_dim
                                 , layer_norm = layer_norm, batch_norm = batch_norm
                                 , num_layers = mlp_num_layers, dropout = dropout
                                 )
        node_model = NodeV1Model(node_dim, edge_dim, global_dim
                                 , layer_norm=layer_norm, batch_norm=batch_norm
                                 , message_mlp_hidden_dim=hidden_dim, message_mlp_num_layers=mlp_num_layers, message_mlp_dropout=dropout
                                 , update_mlp_hidden_dim=hidden_dim, update_mlp_num_layers=mlp_num_layers, update_mlp_dropout=dropout
                                 )
        self.ablation = ablation
        # if 'GAT' in ablation:
        #     self.graph_network = GAT(edge_model, node_model)
        self.graph_network = GraphNetwork(edge_model, node_model, gnn_num_layers, ablation)
        self.attention = MultiheadAttPoolLayer(nodeatt_heads_num, node_feat_num, hidden_dim)
        self.activation = GELU()
        self.dropout_m = nn.Dropout(dropout)
        self.hid2out = MLP(node_dim+edge_dim+node_feat_num, hidden_dim, num_class, mlp_num_layers, dropout, batch_norm=batch_norm, layer_norm=batch_norm)

    def forward(self, x, edge_index, edge_attr, qa_attr, node_batch, num_of_nodes):
        batch_size = qa_attr.size(0)
        edge_batch = node_batch[edge_index[0]]
        max_node_num = torch.max(num_of_nodes).item()

        # worknode transform
        x = self.worknode_transform(x)
        # workrelation transform
        edge_attr = self.workrelation_transform(edge_attr)
        # global_transform
        u = self.global_transform(qa_attr)
        node_vecs, edge_vecs, unnormalized_wts, normalized_wts = self.graph_network(x, edge_index, edge_attr, u, node_batch)

        pooled_edge_vecs = scatter_sum(edge_vecs * normalized_wts, edge_batch, dim=0, dim_size=batch_size)

        evidence_vecs = torch.zeros(batch_size, max_node_num, self.node_dim, device=node_vecs.device)
        j = 0
        for i in range(batch_size):
            visible_num_tuples = min(num_of_nodes[i].item(), max_node_num)
            evidence_vecs[i, : visible_num_tuples, :] = node_vecs[j: j + visible_num_tuples,:]
            j = j + num_of_nodes[i].item()
        evidence_vecs = self.activation(evidence_vecs)
        # pooled_node_vecs
        mask = torch.arange(max_node_num, device=node_vecs.device) >= num_of_nodes.unsqueeze(1)
        mask[mask.all(1), 0] = 0

        pooled_node_vecs, att_scores = self.attention(qa_attr, evidence_vecs, mask)
        embeddings = torch.cat((pooled_edge_vecs, pooled_node_vecs, qa_attr), 1)
        logits = self.hid2out(self.dropout_m(embeddings))

        return logits, [unnormalized_wts, normalized_wts, embeddings]
