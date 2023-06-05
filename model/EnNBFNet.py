from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add

from model.layers import *

def join_path(paths):
    path_str_list = []
    for path in paths:
        edge_list = ['_'.join(list(map(lambda x: str(x), edge))) for edge in path]
        path_str = '_'.join(edge_list)
        path_str_list.append(path_str)
    return ';'.join(path_str_list)

def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample

def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask

def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size

def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        # special case: size < k, pad with the last valid index
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask] # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index

class EdgeModel(torch.nn.Module):
    '''
    def message passing
    '''
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, edge_out_dim
                 ,layer_norm = False, batch_norm = False
                 ,num_layers=2, dropout=0.5
                 ):
        '''
        '''
        super(EdgeModel, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        if self.global_dim is None:
            if self.node_dim is None:
                edge_in_dim = self.edge_dim
            else:
                edge_in_dim = self.node_dim*2 + self.edge_dim
        else:
            if self.node_dim is None:
                edge_in_dim = self.edge_dim + self.global_dim
            else:
                edge_in_dim = self.node_dim*2+self.edge_dim+self.global_dim
        self.edge_mlp = MLP(edge_in_dim, hidden_dim, edge_out_dim,
                            num_layers=num_layers, dropout=dropout, batch_norm=batch_norm, layer_norm=layer_norm)

    def forward(self, src, dest, edge_attr, u=None, edge_batch=None):
        '''
        source, target: [E, F_x], where E is the number of edges.
        edge_attr: [E, F_e]
        u: [B, F_u], where B is the number of graphs.
        batch: [E] with max entry B - 1.
        '''
        if self.global_dim is None:
            if self.node_dim is None:
                out = torch.cat([edge_attr],1)
            else:
                out = torch.cat([src, dest, edge_attr], 1)
        else:
            if self.node_dim is None:
                out = torch.cat([edge_attr, u[edge_batch]], 1)
            else:
                out = torch.cat([src, dest, edge_attr, u[edge_batch]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    '''
    def aggragating and updating
    '''
    def __init__(self, message_func='add', aggregate_func='sum', flow='source_to_target'
                 ,layer_norm = False, batch_norm = False
                 ,dists_emb_dim=128, edge_dim=128, global_dim=128
                 ,message_mlp_hidden_dim=128, message_mlp_num_layers=2, message_mlp_dropout=0.5
                 ,update_mlp_hidden_dim=128, update_mlp_num_layers=2, update_mlp_dropout=0.5):
        super(NodeModel, self).__init__()
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.flow = flow
        self.eps = 10**-8

        if self.message_func == 'concat':
            self.message_mlp = MLP(dists_emb_dim+edge_dim, message_mlp_hidden_dim, dists_emb_dim
                                   ,num_layers=message_mlp_num_layers, dropout=message_mlp_dropout, batch_norm=batch_norm, layer_norm=layer_norm)
        elif self.message_func in ('add','mul'):
            self.message_mlp = None
        else:
            raise ValueError('message_func is out range!', self.message_func)

        # update dists emb
        if self.aggregate_func == 'pna':
            self.update = MLP(dists_emb_dim*13, update_mlp_hidden_dim, dists_emb_dim,num_layers=update_mlp_num_layers, dropout=update_mlp_dropout, batch_norm=batch_norm, layer_norm=layer_norm)
        else:
            self.update = MLP(dists_emb_dim*2, update_mlp_hidden_dim, dists_emb_dim,num_layers=update_mlp_num_layers, dropout=update_mlp_dropout, batch_norm=batch_norm, layer_norm=layer_norm)

    def forward(self, dists_emb, dists_emb_boundary, edge_index, edge_attr, u=None, node_batch=None, edge_weight=None):
        # x: [N, F_x], where N is the number of nodes. original attributed of nodes
        # dists_emb: [N, F_x], where N is the number of nodes. <source_node, node> pair representation
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        if self.flow == 'source_to_target':
            row, col = edge_index #row is src
        elif self.flow == 'target_to_source':
            col, row = edge_index
        else: raise ValueError('flow is out of range!', self.flow)

        if self.message_func == 'concat':
            message = torch.cat([dists_emb[row], edge_attr], dim=1)
            message = self.message_mlp(message)  # message: [E, hidden_dim]; s -> t through w
        elif self.message_func == 'add':
            message = torch.add(dists_emb[row], edge_attr)
        elif self.message_func == 'mul':
            message = dists_emb[row] * edge_attr
        else: pass

        if edge_weight is not None: message = message * torch.unsqueeze(edge_weight,dim=-1) #for computing path importance

        if self.aggregate_func == 'sum':
            update = scatter_sum(message, col, dim=0, dim_size=dists_emb.size(0))
            update = dists_emb_boundary + update
        elif self.aggregate_func == 'mean':
            update = scatter_sum(message, col, dim=0, dim_size=dists_emb.size(0))
            degree = scatter_sum(torch.ones(col.size(0),1, device=message.device), col, dim=0, dim_size=dists_emb.size(0))
            update = (dists_emb_boundary + update)/(degree + 1.0)
        elif self.aggregate_func == 'max':
            update,_ = scatter_max(message, col, dim=0, dim_size=dists_emb.size(0))
            update = torch.maximum(update, dists_emb_boundary)
        elif self.aggregate_func == 'min':
            update,_ = scatter_min(message, col, dim=0, dim_size=dists_emb.size(0))
            update = torch.minimum(update, dists_emb_boundary)
        elif self.aggregate_func == 'pna':
            sum = scatter_sum(message, col, dim=0, dim_size=dists_emb.size(0))
            sum = dists_emb_boundary + sum
            degree = scatter_sum(torch.ones(col.size(0),1,device=message.device), col, dim=0, dim_size=dists_emb.size(0))
            mean = (dists_emb_boundary + sum)/(degree + 1.0)
            sq_sum = scatter_sum(message**2, col, dim=0, dim_size=dists_emb.size(0))
            sq_mean = (dists_emb_boundary**2 + sq_sum)/(degree + 1.0)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            max,_ = scatter_max(message, col, dim=0, dim_size=dists_emb.size(0))
            max = torch.maximum(max, dists_emb_boundary)
            min, _ = scatter_min(message, col, dim=0, dim_size=dists_emb.size(0))
            min = torch.minimum(min, dists_emb_boundary)
            features = torch.cat([mean, max, min, std], dim=-1) #(num_node, hidden_dim*4)
            degree_scale = (degree + 1.0).log() #(num_node,)
            degree_scale = degree_scale/degree_scale.mean()
            degree_scales = torch.cat([torch.ones_like(degree_scale), degree_scale, 1/degree_scale.clamp(min=1e-2)], dim=-1)  #(num_node, 3)
            update = (features.unsqueeze(-1) * degree_scales.unsqueeze(-2)).flatten(-2)  # (num_node,input_dim * 4 * 3)
        else: raise ValueError('aggregate_func is out of range!', self.aggregate_func)

        output = self.update(torch.cat([update, dists_emb],dim=1))
        return output, edge_weight

class EnNBFNet(torch.nn.Module):
    def __init__(self, num_layers=3, edge_model_share=False, node_model_share=False
                 ,ablation = []
                 ,message_func='add', aggregate_func='sum', flow='source_to_target', short_cut=False
                 ,node_dim=128, dists_emb_dim=128, edge_dim=128, global_dim=128, hidden_dim=128
                 ,mlp_num_layers=2, layer_norm=False, batch_norm=False, dropout=0.5
                 ):
        super(EnNBFNet, self).__init__()
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.dists_emb_dim = dists_emb_dim
        self.flow = flow
        self.short_cut = short_cut
        if self.short_cut: print('short cut')
        else: print('not short cut')
        self.ablation = ablation
        print('ablation: ',self.ablation)

        self.query = nn.Embedding(1,self.dists_emb_dim) #nn.parameter.Parameter(torch.empty((1, self.dists_emb_dim),dtype=torch.float32))
        #nn.Embedding(1,self.dists_emb_dim) #boundary

        self.edge_model_share = edge_model_share
        if 'no_globaldim' in self.ablation: edge_model_global_dim = None
        else: edge_model_global_dim = global_dim
        if 'no_nodedim' in self.ablation: edge_model_node_dim =None
        else: edge_model_node_dim = node_dim

        if self.edge_model_share:
            print('edge_model share param')
            self.edge_model = EdgeModel(edge_model_node_dim, edge_dim, edge_model_global_dim, hidden_dim, edge_out_dim=edge_dim, layer_norm=layer_norm, batch_norm=batch_norm, num_layers=mlp_num_layers, dropout=dropout)
        else:
            print('edge_model not share param')
            self.edge_model = nn.ModuleList()
            for layer in range(num_layers):
                self.edge_model.append(EdgeModel(edge_model_node_dim, edge_dim, edge_model_global_dim, hidden_dim, edge_out_dim=edge_dim, layer_norm=layer_norm, batch_norm=batch_norm, num_layers=mlp_num_layers, dropout=dropout))

        self.node_model_share = node_model_share
        if self.node_model_share:
            print('node_model share param')
            self.node_model = NodeModel(message_func, aggregate_func, flow
                                        , layer_norm, batch_norm
                                        , dists_emb_dim, edge_dim, global_dim
                                        , message_mlp_hidden_dim=hidden_dim, message_mlp_num_layers=mlp_num_layers, message_mlp_dropout=dropout
                                        , update_mlp_hidden_dim=hidden_dim, update_mlp_num_layers=mlp_num_layers, update_mlp_dropout=dropout)
        else:
            print('node_model not share param')
            self.node_model = nn.ModuleList()
            for layer in range(num_layers):
                self.node_model.append(NodeModel(message_func, aggregate_func, flow
                                                 , layer_norm, batch_norm
                                                 , dists_emb_dim, edge_dim, global_dim
                                                 , message_mlp_hidden_dim=hidden_dim, message_mlp_num_layers=mlp_num_layers, message_mlp_dropout=dropout
                                                 , update_mlp_hidden_dim=hidden_dim, update_mlp_num_layers=mlp_num_layers, update_mlp_dropout=dropout)
                                       )

    def __get_modelitem(self, model, idx):
        if isinstance(model,torch.nn.modules.container.ModuleList):
            return model[idx]
        else: return model
    def forward(self, x, x_source_vec, edge_index, edge_attr, u, node_batch, x_dist=None, edge_weight=None):
        '''
        Args:
            x: [N, node_dim]
            x_source_vec: [N, 1] element=0/1, 1=anchor;
            edge_index:
            edge_attr:
            u:
            node_batch:

        Returns:

        '''
        boundary = torch.zeros(x.size(0), self.dists_emb_dim, device=x_source_vec.device)
        if x_dist is None:
            boundary = boundary + x_source_vec * self.query.weight # boundary
        else:
            boundary = boundary + x_source_vec * self.query.weight + x_dist # boundary

        layer_input = boundary

        # edge_attr_list = [edge_attr,]
        edge_attr_org = edge_attr
        dists_emb_list = []
        edge_weight_list = []
        for layer in range(self.num_layers):
            edge_model = self.__get_modelitem(self.edge_model,layer)
            node_model = self.__get_modelitem(self.node_model,layer)
            if self.flow == 'source_to_target':
                row, col = edge_index  # row is src
            elif self.flow == 'target_to_source':
                col, row = edge_index
            else:
                raise ValueError('flow is out of range!', self.flow)
            edge_batch = node_batch[row] #no matter row/col
            # 更新边信息
            edge_attr = edge_model(x[row], x[col], edge_attr_org, u, edge_batch)
            # edge_attr_list.append(edge_attr)
            # update node dists_emb
            if 'exppath_sep' in self.ablation: # for computing path importance
                edge_weight = edge_weight.clone().requires_grad_()
            layer_output, edge_weight = node_model(layer_input, boundary, edge_index, edge_attr, u=u, node_batch=node_batch, edge_weight=edge_weight)

            if self.short_cut:
                layer_output = layer_output + layer_input
            else: pass
            dists_emb_list.append(layer_output)
            edge_weight_list.append(edge_weight)
            layer_input = layer_output

        return dists_emb_list, edge_weight_list

class EnNBFModel(nn.Module):
    '''
        model from Net
    '''
    def __init__(self, gnn_num_layers=3, edge_model_share=False, node_model_share=False, num_class=2, cate_choice='lastgnn_query_qa'
                 , ablation = []
                 , message_func='add', aggregate_func='sum', flow='bi_direction', short_cut=False
                 , worknode_feat_num=None, workrelation_feat_num=None, node_feat_num=None
                 , node_dim=128, dists_emb_dim=128, edge_dim=128, global_dim=128, hidden_dim=128
                 , mlp_num_layers=2, layer_norm=False, batch_norm=False, dropout=0.5
                 ):
        super(EnNBFModel, self).__init__()
        self.gnn_num_layers = gnn_num_layers
        self.cate_choice = cate_choice
        self.flow = flow
        self.short_cut = short_cut
        # node transform
        self.worknode_transform = nn.Linear(worknode_feat_num, node_dim)
        # edge transform
        self.workrelation_transform = nn.Linear(workrelation_feat_num, edge_dim)
        # global transform
        if 'globallinear' in ablation:
            self.global_transform = nn.Linear(node_feat_num, global_dim)
        else:
            self.global_transform = MLP(node_feat_num, hidden_dim, global_dim, num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm, layer_norm=layer_norm)

        if self.flow == 'bi_direction':
            nbfnet_output_channel = 2
            self.gnn_source2target = EnNBFNet(gnn_num_layers, edge_model_share, node_model_share
                                              , ablation
                                              , message_func, aggregate_func, 'source_to_target', short_cut
                                              , node_dim, dists_emb_dim, edge_dim, global_dim, hidden_dim
                                              , mlp_num_layers, layer_norm, batch_norm, dropout
                                              )
            self.gnn_target2source = EnNBFNet(gnn_num_layers, edge_model_share, node_model_share
                                              , ablation
                                              , message_func, aggregate_func, 'target_to_source', short_cut
                                              , node_dim, dists_emb_dim, edge_dim, global_dim, hidden_dim
                                              , mlp_num_layers, layer_norm, batch_norm, dropout
                                              )
        else:
            nbfnet_output_channel = 1
            self.gnn = EnNBFNet(gnn_num_layers, edge_model_share, node_model_share
                                , ablation
                                , message_func, aggregate_func, self.flow, short_cut
                                , node_dim, dists_emb_dim, edge_dim, global_dim, hidden_dim
                                , mlp_num_layers, layer_norm, batch_norm, dropout
                                )

        if cate_choice == 'lastgnn_query_qa':
            self.cate_model = MLP(dists_emb_dim*nbfnet_output_channel*2+node_feat_num, hidden_dim, num_class,
                                  num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm,
                                  layer_norm=layer_norm)
        elif cate_choice == 'lastgnn_qa':
            self.cate_model = MLP(dists_emb_dim*nbfnet_output_channel+node_feat_num, hidden_dim, num_class,
                                  num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm,
                                  layer_norm=layer_norm)
        else: raise ValueError('cate choice is out of range!', cate_choice)

    def forward(self, x, x_source_vec, x_target_vec, edge_index, edge_attr, qa_attr, node_batch, edge_weight=None):
        batch_size = qa_attr.size(0)

        # worknode transform
        x = self.worknode_transform(x)
        # workrelation transform
        edge_attr = self.workrelation_transform(edge_attr)
        # global_transform
        u = self.global_transform(qa_attr) #xixi_todo detach u = qa_attr.clone().detach()

        if self.flow == 'bi_direction':
            dists_emb_list_1, edge_weight_list_1 = self.gnn_source2target(x, x_source_vec, edge_index, edge_attr, u, node_batch, edge_weight=edge_weight)
            dists_emb_list_2, edge_weight_list_2 = self.gnn_target2source(x, x_target_vec, edge_index, edge_attr, u, node_batch, edge_weight=edge_weight)
            query_output = torch.cat([self.gnn_source2target.query.weight,self.gnn_target2source.query.weight], dim=-1).expand(batch_size, -1)
            gnn_output_1 = scatter_sum(dists_emb_list_1[-1] * x_target_vec, node_batch, dim=0, dim_size=batch_size)
            gnn_output_2 = scatter_sum(dists_emb_list_2[-1] * x_source_vec, node_batch, dim=0, dim_size=batch_size)
            gnn_output = torch.cat([gnn_output_1,gnn_output_2], dim=-1)

            edge_weight_list = [edge_weight_list_1,edge_weight_list_2]
        elif self.flow == 'source_to_target':
            dists_emb_list, edge_weight_list = self.gnn(x, x_source_vec, edge_index, edge_attr, u, node_batch, edge_weight=edge_weight)
            assert len(dists_emb_list) == self.gnn_num_layers
            query_output = self.gnn.query.weight.expand(batch_size,-1)
            gnn_output = scatter_sum(dists_emb_list[-1] * x_target_vec, node_batch, dim=0, dim_size=batch_size)
        elif self.flow == 'target_to_source':
            dists_emb_list, edge_weight_list = self.gnn(x, x_target_vec, edge_index, edge_attr, u, node_batch, edge_weight=edge_weight)
            assert len(dists_emb_list) == self.gnn_num_layers
            query_output = self.gnn.query.weight.expand(batch_size,-1)
            gnn_output = scatter_sum(dists_emb_list[-1] * x_source_vec, node_batch, dim=0, dim_size=batch_size)
        else: pass
        if self.cate_choice == 'lastgnn_query_qa':
            output = torch.cat([gnn_output, query_output, qa_attr], dim=-1)
            logits = self.cate_model(output)
        elif self.cate_choice == 'lastgnn_qa':
            output = torch.cat([gnn_output, qa_attr], dim=-1)
            logits = self.cate_model(output)
        return logits, output, edge_weight_list

@torch.no_grad()
def beam_search_distance(edge_index, edge_type, num_of_nodes, edge_grads, h_index, t_index, num_beam=10):
    # beam search the top-k distance from h to t (and to every other node)
    num_nodes = torch.sum(num_of_nodes)
    input = torch.full((num_nodes, num_beam), float("-inf"), device=num_of_nodes.device)
    input[h_index, 0] = 0
    edge_mask = edge_index[0, :] != t_index
    distances = []
    back_edges = []
    for edge_grad in edge_grads:
        # we don't allow any path goes out of t once it arrives at t
        node_in, node_out = edge_index[:, edge_mask]
        relation = edge_type[edge_mask]
        edge_grad = edge_grad[edge_mask]

        message = input[node_in] + edge_grad.unsqueeze(-1)  # (num_edges, num_beam)
        # (num_edges, num_beam, 3)
        msg_source = torch.stack([node_in, node_out, relation], dim=-1).unsqueeze(1).expand(-1, num_beam, -1)

        # (num_edges, num_beam)
        is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                       (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
        # pick the first occurrence as the ranking in the previous node's beam
        # this makes deduplication easier later
        # and store it in msg_source
        is_duplicate = is_duplicate.float() - \
                       torch.arange(num_beam, dtype=torch.float, device=message.device) / (num_beam + 1)
        prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
        msg_source = torch.cat([msg_source, prev_rank], dim=-1)  # (num_edges, num_beam, 4)

        node_out, order = node_out.sort()
        node_out_set = torch.unique(node_out)
        # sort messages w.r.t. node_out
        message = message[order].flatten()  # (num_edges * num_beam)
        msg_source = msg_source[order].flatten(0, -2)  # (num_edges * num_beam, 4)
        size = node_out.bincount(minlength=num_nodes)
        msg2out = size_to_index(size[node_out_set] * num_beam)
        # deduplicate messages that are from the same source and the same beam
        is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
        is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate])
        message = message[~is_duplicate]
        msg_source = msg_source[~is_duplicate]
        msg2out = msg2out[~is_duplicate]
        size = msg2out.bincount(minlength=len(node_out_set))

        if not torch.isinf(message).all():
            # take the topk messages from the neighborhood
            # distance: (len(node_out_set) * num_beam)
            distance, rel_index = scatter_topk(message, size, k=num_beam)
            abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
            # store msg_source for backtracking
            back_edge = msg_source[abs_index]  # (len(node_out_set) * num_beam, 4)
            distance = distance.view(len(node_out_set), num_beam)
            back_edge = back_edge.view(len(node_out_set), num_beam, 4)
            # scatter distance / back_edge back to all nodes
            distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_nodes)  # (num_nodes, num_beam)
            back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_nodes)  # (num_nodes, num_beam, 4)
        else:
            distance = torch.full((num_nodes, num_beam), float("-inf"), device=message.device)
            back_edge = torch.zeros(num_nodes, num_beam, 4, dtype=torch.long, device=message.device)

        distances.append(distance)
        back_edges.append(back_edge)
        input = distance

    return distances, back_edges

def topk_average_length(distances, back_edges, t_index, k=10):
    # backtrack distances and back_edges to generate the paths
    paths = []
    average_lengths = []
    for i in range(len(distances)):
        distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
        back_edge = back_edges[i][t_index].flatten(0, -2)[order]
        for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
            if d == float("-inf"):
                break
            path = [(h, t, r)]
            for j in range(i - 1, -1, -1):
                h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                path.append((h, t, r))
            paths.append(path[::-1])
            average_lengths.append(d / len(path))

    if paths:
        average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

    return paths, average_lengths

class MLPwithkgModel(torch.nn.Module):
    def __init__(self, num_class = 2, nodeatt_heads_num=2
                 , worknode_feat_num=None, workrelation_feat_num=None, node_feat_num=None
                 , node_dim=128, edge_dim=128, global_dim=128, hidden_dim=128
                 , mlp_num_layers=2, layer_norm=False, batch_norm=False, dropout=0.5
                 ):
        super(MLPwithkgModel, self).__init__()
        self.worknode_feat_num = worknode_feat_num
        self.node_dim = node_dim
        # node transform
        self.worknode_transform = nn.Linear(worknode_feat_num, node_dim)
        # # edge transform
        # self.workrelation_transform = nn.Linear(workrelation_feat_num, edge_dim)
        # global transform
        self.global_transform = MLP(node_feat_num, hidden_dim, global_dim, num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm, layer_norm=layer_norm)
        self.activation = GELU()
        self.attention = MultiheadAttPoolLayer(nodeatt_heads_num, global_dim, node_dim)
        self.cate_model = MLP(node_dim+node_feat_num, hidden_dim, num_class, num_layers=mlp_num_layers, dropout=dropout, batch_norm=batch_norm, layer_norm=layer_norm)

    def forward(self, x, x_source_vec, x_target_vec, edge_index, edge_attr, qa_attr, node_batch, num_of_nodes):
        batch_size = num_of_nodes.shape[0]
        max_node_num = torch.max(num_of_nodes).item()
        # node transform
        x = self.worknode_transform(x)
        # global_transform
        u = self.global_transform(qa_attr)  # xixi_todo detach u = qa_attr.clone().detach()

        # evidence_vecs for node aggregation with attention
        mask = torch.arange(max_node_num, device=x.device) >= num_of_nodes.unsqueeze(1) #(bz, max_node_num)
        # mask[mask.all(1), 0] = 0
        evidence_vecs = torch.zeros(batch_size, max_node_num, self.node_dim, device=x.device)
        j = 0
        for i in range(batch_size):
            visible_num_tuples = min(num_of_nodes[i].item(), max_node_num) # num of graphs
            evidence_vecs[i, : visible_num_tuples, :] = x[j: j + visible_num_tuples, :]
            mask[i,:visible_num_tuples] = torch.logical_or(mask[i,:visible_num_tuples], torch.logical_or(x_source_vec[j: j + visible_num_tuples,0]>0.5, x_target_vec[j: j + visible_num_tuples,0]>0.5))
            j = j + num_of_nodes[i].item()
        evidence_vecs = self.activation(evidence_vecs)

        # attention(q,k,mask),  evidence_vecs = k， pooling with attention
        pooled_node_vecs, att_scores = self.attention(u, evidence_vecs, mask)
        embeddings = torch.cat((pooled_node_vecs, qa_attr), 1)

        logits = self.cate_model(embeddings)
        return logits
