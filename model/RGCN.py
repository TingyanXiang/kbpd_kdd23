from torch_geometric.nn import RGCNConv

from model.layers import *

class RGCN(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_class,
                 num_relations,
                 metadata=None,
                 n_bases=8,
                 n_layers=2,
                 out_layer_num=0,
                 dropout_rate=0.4,
                 default_label=-1):
        super().__init__()
        self.default_label = default_label
        self.dropout_rate = dropout_rate

        self.input_layer = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=n_bases))
        if out_layer_num == 0:
            self.out_layer = nn.Linear(hidden_channels, num_class)
        elif out_layer_num >= 1:
            self.out_layer = MLP(hidden_channels, hidden_channels, num_class, out_layer_num, dropout=dropout_rate,
                                 batch_norm=False,
                                 init_last_layer_bias_to_zero=False, layer_norm=True, activation='gelu', bias=True)
        else:
            raise ValueError('out_layer_num error!', out_layer_num)

    def forward(self, data):
        # root_index = data['root_index']
        # to_homogeneous()
        data = data.to_homogeneous()
        # get input
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        # forward
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.input_layer(x).relu_()
        x = F.normalize(x, dim=-1, p=2)

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = conv(x, edge_index, edge_type).relu_()
            x = F.normalize(x, dim=-1, p=2)

        x = self.out_layer(x)

        # return id, out and label
        root_index = (data.y != self.default_label)
        Y = data.y[root_index]
        out = x[root_index]
        org_ids = data.org_ids[root_index]

        return out, Y, org_ids

class RGCN_wokgmp(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_class,
                 num_relations,
                 metadata=None,
                 n_bases=8,
                 n_layers=2,
                 out_layer_num=0,
                 dropout_rate=0.4,
                 default_label=-1):
        super().__init__()
        self.default_label = default_label
        self.dropout_rate = dropout_rate

        self.input_layer = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=n_bases))
        if out_layer_num == 0:
            self.out_layer = nn.Linear(hidden_channels + 256, num_class)
        elif out_layer_num >= 1:
            self.out_layer = MLP(hidden_channels + 256, hidden_channels, num_class, out_layer_num, dropout=dropout_rate,
                                 batch_norm=False,
                                 init_last_layer_bias_to_zero=False, layer_norm=True, activation='gelu', bias=True)
        else:
            raise ValueError('out_layer_num error!', out_layer_num)

    def forward(self, data):
        # to_homogeneous()
        data = data.to_homogeneous()
        # get input
        row_x = data.x
        x = row_x[:, 256:]
        kbpd_emb = row_x[:, :256]
        edge_index = data.edge_index
        edge_type = data.edge_type

        # forward
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.input_layer(x).relu_()
        x = F.normalize(x, dim=-1, p=2)

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = conv(x, edge_index, edge_type).relu_()
            x = F.normalize(x, dim=-1, p=2)

        x = torch.cat([x, kbpd_emb], dim=1)
        x = self.out_layer(x)

        # return out and label
        root_index = (data.y != self.default_label)
        Y = data.y[root_index]
        out = x[root_index]

        return out, Y
