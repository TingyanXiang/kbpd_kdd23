import torch
from model.layers import *

class MLPModel(torch.nn.Module):
    def __init__(self,num_class = 2, cate_choice='item_kghit'
                 ,node_feat_num = None, item_hidden_dim=128, node_dim=128
                 ,cate_hidden_dim=64
                 ,mlp_num_layers=2, layer_norm=False, batch_norm=False, dropout=0.5
                 ):
        super(MLPModel, self).__init__()
        self.cate_choice = cate_choice
        self.item_mlp = MLP(node_feat_num, item_hidden_dim, node_dim, num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm, layer_norm=layer_norm)
        if self.cate_choice == 'item_kghit':
            self.cate_mlp = MLP(node_dim+1, cate_hidden_dim, num_class, num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm, layer_norm=layer_norm)
        elif self.cate_choice == 'item':
            self.cate_mlp = MLP(node_dim, cate_hidden_dim, num_class, num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm, layer_norm=layer_norm)
        else: raise ValueError('cate_choice error!!!', cate_choice)
    def forward(self, u, hit_vec):
        output1 = self.item_mlp(u)
        if self.cate_choice == 'item_kghit':
            output2 = torch.cat([output1,hit_vec],dim=-1)
        else:
            output2 = output1
        logits = self.cate_mlp(output2)
        return logits