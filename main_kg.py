# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from tqdm import tqdm
import graphlearn as gl
import graphlearn.python.nn.pytorch as thg
from torch import autograd
from torch_geometric.data import Data
from utils.gl_io_utils import HeteGraph
from model.EnNBFNet import *
from model.GraIL import *
from utils.utils import bool_flag, printWithTimes, save_model, load_model

argv = sys.argv[0:]
# import and process config
from config.config_kg import config, graph_config
cur_path = os.getcwd() #directory when running; by defualt, =directory where cloning this repository.
print('cur_path: ', cur_path)
if graph_config.get('data_root_path', None) is None:
    graph_config['data_root_path'] = os.path.join(cur_path, 'kbpd_kdd23_open_code_v2', 'data')
if config.get('save_root_path', None) is None:
    config['save_root_path'] = os.path.join(cur_path, 'saved_models')
if len(argv)>1:
    config['mode'] = argv[1]
print('config: ', config)

# running environment setting
if torch.cuda.is_available(): config['device'] = 'cuda'
else: config['device'] = 'cpu'
device = torch.device(config['device'])
world_size = 1
rank = 0
# torch distribution setting
if bool_flag(config.get('ddp',False)):
    if torch.cuda.is_available():
        torch.distributed.init_process_group('nccl')
    else:
        torch.distributed.init_process_group('gloo')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

# define model
print('build model...')
if config['model_choice'] == 'mlpkg':
    model = MLPwithkgModel(num_class = config['num_class']
                     , nodeatt_heads_num = 2
                     , worknode_feat_num = config['worknode_feat_num']
                     , workrelation_feat_num = config['workrelation_feat_num']
                     , node_feat_num = config['node_feat_num']
                     , node_dim = config['gnn_hidden_dim']
                     , edge_dim=config['gnn_hidden_dim']
                     , global_dim=config['gnn_hidden_dim']
                     , hidden_dim=config['gnn_hidden_dim']
                     , mlp_num_layers=2
                     , layer_norm = bool_flag(config['layer_norm'])
                     , batch_norm = bool_flag(config['batch_norm'])
                     , dropout = config['dropout_rate']
                    ).to(device)
elif config['model_choice'] == 'grail':
    model = GrailModel(gnn_num_layers=config['gnn_num_layers']
                      , nodeatt_heads_num=2
                      , num_class=config['num_class']
                      , ablation = config['ablation']
                      , worknode_feat_num = config['worknode_feat_num']
                      , workrelation_feat_num=config['workrelation_feat_num']
                      , node_feat_num=config['node_feat_num']
                      , node_dim=config['gnn_hidden_dim']
                      , edge_dim=config['gnn_hidden_dim']
                      , global_dim=config['gnn_hidden_dim']
                      , hidden_dim=config['gnn_hidden_dim']
                      , mlp_num_layers=2
                      , layer_norm=bool_flag(config['layer_norm'])
                      , batch_norm=bool_flag(config['batch_norm'])
                      , dropout=config['dropout_rate']
                      ).to(device)
elif config['model_choice'] == 'grailv1':
    model = GrailV1Model(gnn_num_layers=config['gnn_num_layers']
                      , nodeatt_heads_num=2
                      , num_class = config['num_class']
                      , ablation=config['ablation']
                      , worknode_feat_num = config['worknode_feat_num']
                      , workrelation_feat_num=config['workrelation_feat_num']
                      , node_feat_num=config['node_feat_num']
                      , node_dim=config['gnn_hidden_dim']
                      , edge_dim=config['gnn_hidden_dim']
                      , global_dim=config['gnn_hidden_dim']
                      , hidden_dim=config['gnn_hidden_dim']
                      , mlp_num_layers=2
                      , layer_norm=bool_flag(config['layer_norm'])
                      , batch_norm=bool_flag(config['batch_norm'])
                      , dropout=config['dropout_rate']
                      ).to(device)
else:
    model = EnNBFModel(gnn_num_layers=config['gnn_num_layers']
                       , edge_model_share = bool_flag(config['edge_model_share'])
                       , node_model_share = bool_flag(config['node_model_share'])
                       , num_class =config['num_class']
                       , cate_choice = config['cate_choice']
                       , ablation = config['ablation']
                       , message_func = config['message_func']
                       , aggregate_func = config['aggregate_func']
                       , flow = config['flow']
                       , short_cut = bool_flag(config['short_cut'])
                       , worknode_feat_num =config['worknode_feat_num']
                       , workrelation_feat_num = config['workrelation_feat_num']
                       , node_feat_num = config['node_feat_num']
                       , node_dim = config['gnn_hidden_dim']
                       , dists_emb_dim =config['gnn_hidden_dim']
                       , edge_dim = config['gnn_hidden_dim']
                       , global_dim = config['gnn_hidden_dim']
                       , hidden_dim = config['gnn_hidden_dim']
                       , mlp_num_layers= config['mlp_num_layers']
                       , layer_norm = bool_flag(config['layer_norm'])
                       , batch_norm = bool_flag(config['batch_norm'])
                       , dropout=config['dropout_rate']
                       ).to(device)
print('build model done.')
if world_size > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
print(model)

# choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.0)

# init graph through gl
print("init graph...")
gl.set_tape_capacity(1)
hete_graph = HeteGraph(graph_config)
gl.set_tracker_mode(0)
hete_graph.init_gl(rank, world_size)
print("init graph done.")

def homo_graph_parser(graph_attrs, workgraph_edgestr_len=6):
    '''
        transform subgraph info from array-format to pygdata-format
        param
        >> graph_attrs: 1d-array of int = (spo_cnt, s1, s1_type, p1, p1_weight, o1, o1_type, ..., sn, sn_type, pn, pn_weight, on, on_type), and p_weight is optional.
        >> workgraph_edgestr_len: number of elements for each edge.
        return
        >> edge_index, edge_attr, edge_type, edge_weight, node_ids, node_attrs, node_types, x_source_vec, x_target_vec, row_node_ids, col_node_ids
    '''

    spo_cnt = graph_attrs[0]
    if int(spo_cnt) == 0:  #if no kg, add one 0-0 edge
        edge = np.zeros([1,workgraph_edgestr_len], dtype=np.int8) + (-1)
    else:
        edge = graph_attrs[1:].reshape(-1, workgraph_edgestr_len)[:spo_cnt]  #(source_id, source_node_type, relation_id, weight, target_id, target_node_type)
    row_node_ids = edge[:, 0].tolist()
    if workgraph_edgestr_len == 5:
        col_node_ids = edge[:, 3].tolist()
    elif workgraph_edgestr_len == 6:
        col_node_ids = edge[:, 4].tolist()
    else:
        raise ValueError('workgraph_edgestr_len Error!!!', workgraph_edgestr_len)

    # reindex
    node_ids = np.array(list(set(row_node_ids) | set(col_node_ids)))  #localid2id
    id2localid = dict()
    for node_idx, node_id in enumerate(node_ids):
        id2localid[node_id] = node_idx
    row = [id2localid[node_id] for node_id in row_node_ids]
    col = [id2localid[node_id] for node_id in col_node_ids]
    edge_index = np.stack([np.array(row), np.array(col)], axis=0)

    edge_type = np.array(edge[:, 2].tolist())
    workrelation = hete_graph.g.get_nodes('workrelation', edge_type)
    edge_attr = workrelation.float_attrs
    if workgraph_edgestr_len == 6: edge_weight = np.array(edge[:, 3].tolist(),dtype=np.float32)/100000000.0
    else: edge_weight = None

    worknode = hete_graph.g.get_nodes('worknode', node_ids)
    node_attrs = worknode.float_attrs
    node_types = worknode.labels
    x_source_vec = np.where(node_types==graph_config['source_node_type'], np.ones_like(node_types), np.zeros_like(node_types)) if spo_cnt>0.5 else np.zeros_like(node_types) # flag source_node
    x_target_vec = np.where(node_types==graph_config['target_node_type'], np.ones_like(node_types), np.zeros_like(node_types)) if spo_cnt>0.5 else np.zeros_like(node_types) # flag target_node

    return edge_index, edge_attr, edge_type, edge_weight, node_ids, node_attrs, node_types, x_source_vec, x_target_vec, row_node_ids, col_node_ids

def induce_func(data_dict):
    '''
    transform batch data in dict to pyg data list.
    params
    >> data_dict: batch data in dict generated by graph-learn query
        data_dict['src']
        has attrs：
            ids
            int_attrs
            float_attrs
            string_attrs
            weights
            labels
    return
    >> subgraph_list: list of subgraphs in pyg data format
    '''
    src = data_dict['src'] #gl node object
    subgraph_list = []
    for i in range(src.ids.size):
        qa_id = src.ids[i]
        workgraph_attrs = src.int_attrs[i] #(bz, dim of int_attrs) -> (dim of int_attrs)
        qa_attr = src.float_attrs[i] #(node_feat_num,)

        data = Data()
        label = np.expand_dims(src.labels[i], axis=0)
        data.y = torch.from_numpy(label).to(torch.long)

        data.qa_id = torch.from_numpy(np.expand_dims(qa_id, axis=0)).to(torch.long)
        data.qa_attr = torch.from_numpy(np.expand_dims(qa_attr, axis=0))

        # subgraph的信息
        edge_index, edge_attr, edge_type, edge_weight, node_ids, node_attrs, node_types, x_source_vec, x_target_vec, row_node_ids, col_node_ids = homo_graph_parser(workgraph_attrs, workgraph_edgestr_len=config['workgraph_edgestr_len'])
        data.edge_index = torch.from_numpy(edge_index).to(torch.long)
        data.edge_attr = torch.from_numpy(edge_attr)
        if edge_weight is not None: data.edge_weight = torch.from_numpy(edge_weight)

        data.id = torch.from_numpy(node_ids).to(torch.long) #localid2id: idx=localid value=id;
        data.x = torch.from_numpy(node_attrs)
        data.node_type = torch.from_numpy(node_types).to(torch.long)
        data.edge_type = torch.from_numpy(edge_type).to(torch.long)
        data.x_source_vec = torch.from_numpy(x_source_vec).to(torch.long)
        data.x_target_vec =  torch.from_numpy(x_target_vec).to(torch.long)

        data.num_of_nodes = torch.LongTensor([data.x.shape[0]])
        data.num_of_edges = torch.LongTensor([data.edge_index.shape[1]])

        data.row_node_ids = row_node_ids
        data.col_node_ids = col_node_ids

        subgraph_list.append(data)

    return subgraph_list

mask = graph_config['tables_dict']['sample']['kg_sample_table']['mask']
print('mask: ', mask)
query = hete_graph.feed_next_nodeonly(config['batch_size'], node_type='samplenode', mask=mask)
dataset = thg.Dataset(query, window=5, induce_func=induce_func)
count_per_server = hete_graph.g.server_get_stats()[gl.get_mask_type('samplenode', mask)]
print('sample count per server: ', count_per_server)
num_step_per_epoch = min(count_per_server) // config['batch_size']
print('num_step_per_epoch being set to: ' + str(num_step_per_epoch))

if config['mode'] == 'train':
    print("train ...")
    train_loader = thg.PyGDataLoader(dataset, length=num_step_per_epoch)
    model.train()

    loss_list = []
    for epoch in range(config['num_epoch']):
        step = 0
        for i, data in tqdm(enumerate(train_loader)):
            step += 1
            data = data.to(device)
            qa_attr = data.qa_attr # (bz, attr_dim)
            qa_id = data.qa_id
            id = data.id # (node_num of all subgraphs,)
            node_type = data.node_type
            x = data.x
            edge_index = data.edge_index #(edge_num of all subgraphs, 2)
            edge_attr = data.edge_attr
            edge_type = data.edge_type
            if 'weight' in config['ablation']: edge_weight = data.edge_weight
            else: edge_weight = None
            node_batch = data.batch #(node_num of all subgraphs,[0-(bz-1)])
            num_of_nodes = data.num_of_nodes
            num_of_edges = data.num_of_edges
            x_source_vec = torch.unsqueeze(data.x_source_vec,dim=-1) #(num_of_nodes, 1)
            x_target_vec = torch.unsqueeze(data.x_target_vec,dim=-1)
            row_node_ids = data.row_node_ids
            col_node_ids = data.col_node_ids

            label = data.y.cpu().detach().numpy()
            if np.min(label) < 0:
                print("qa_id:", qa_id)
                print("id:", id)
                print("label:", label)
                raise ValueError("Invalid label in training dataset!!!")

            if config['model_choice'] == 'mlpkg':
                logits = model(x, x_source_vec, x_target_vec, edge_index, edge_attr, qa_attr, node_batch, num_of_nodes)
            elif config['model_choice'] in ('kbpd', 'kbpdv1'):
                logits, _ = model(x, edge_index, edge_attr, qa_attr, node_batch, num_of_nodes)
            elif config['model_choice'] == 'ennbf':
                logits, _ = model(x, x_source_vec, x_target_vec, edge_index, edge_attr, qa_attr, node_batch, num_of_nodes)
            else:
                logits, output, _ = model(x, x_source_vec, x_target_vec, edge_index, edge_attr, qa_attr, node_batch, edge_weight)

            loss = criterion(logits, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_labels = data.y

            loss_list = loss_list[-10:]+[loss.item(),]
            if (epoch==0 and step<=50) or step % config['num_step_print'] == 0:
                print('batch_labels:', data.y.shape, np.sum(data.y.cpu().detach().numpy()))
                printWithTimes('Epoch:[%d/%d]:, Step:[%d/%d] Loss: %.4f avgLoss: %.4f' % (epoch, config['num_epoch'], step, num_step_per_epoch, loss.item(), np.mean(loss_list)))
    # save model
    save_model(model, config['save_root_path'], config['model_name'])

if config['mode'] == 'test':
    test_loader = thg.PyGDataLoader(dataset)
    writer = open(os.path.join(config['save_root_path'], config['model_name'],'output'),'w')
    if bool_flag(config['out_emb']):
        writer.write("\t".join(['node_id:int64', 'label:int64', 'attrs:string'])+"\n")

    #load model
    model = load_model(model, config['save_root_path'], config['model_name'])
    model.eval()

    for epoch in range(1):
        step = 0
        for i, data in tqdm(enumerate(test_loader)):
            data = data.to(device)
            qa_attr = data.qa_attr  # (bz, attr_dim)
            qa_id = data.qa_id
            id = data.id  # (node_num of all subgraphs,)
            node_type = data.node_type
            x = data.x
            edge_index = data.edge_index  # (edge_num of all subgraphs, 2)
            edge_attr = data.edge_attr
            edge_type = data.edge_type
            if 'weight' in config['ablation']:
                edge_weight = data.edge_weight
            elif 'exppath_sep' in config['ablation']:
                edge_weight = torch.ones(torch.sum(data.num_of_edges), device=x.device)
            else:
                edge_weight = None
            node_batch = data.batch  # (node_num of all subgraphs,[0-(bz-1)])
            num_of_nodes = data.num_of_nodes
            num_of_edges = data.num_of_edges
            x_source_vec = torch.unsqueeze(data.x_source_vec,dim=-1)
            x_target_vec = torch.unsqueeze(data.x_target_vec,dim=-1)
            row_node_ids = data.row_node_ids
            col_node_ids = data.col_node_ids

            if config['model_choice'] == 'mlpkg':
                logits = model(x, x_source_vec, x_target_vec, edge_index, edge_attr, qa_attr, node_batch, num_of_nodes)
            elif config['model_choice'] in ('kbpd', 'kbpdv1'):
                logits, _ = model(x, edge_index, edge_attr, qa_attr, node_batch, num_of_nodes)
            elif config['model_choice'] == 'ennbf':
                logits, _ = model(x, x_source_vec, x_target_vec, edge_index, edge_attr, qa_attr, node_batch, num_of_nodes)
            else:
                logits, embeddings, edge_weight_list = model(x, x_source_vec, x_target_vec, edge_index, edge_attr, qa_attr, node_batch, edge_weight)

            batch_labels = data.y
            values = []
            ids = [int(qid) for qid in qa_id]
            labels = [int(label) for label in batch_labels]

            predictions = []
            for p in logits.cpu().detach().numpy():
                predictions.append(','.join([str(e) for e in p]))

            if not bool_flag(config['out_emb']):
                out_embeddings = [None, ] * len(ids)
            else:
                out_embeddings = []
                for emb in embeddings.cpu().detach().numpy():
                    out_embeddings.append(','.join(list(map(lambda x: str(x), emb))))

            # explain path
            if 'exppath_sep' in config['ablation']:
                assert config['batch_size'] == 1
                score = nn.functional.softmax(logits,dim=-1)[:,1]
                id = id.cpu().numpy()
                edge_weight_list1, edge_weight_list2 = edge_weight_list
                edge_weight_list_union = edge_weight_list1+edge_weight_list2
                edge_weight_list_union_len = len(edge_weight_list_union)
                edge_grads_two = autograd.grad(score, edge_weight_list_union)
                edge_grads_1, edge_grads_2 = edge_grads_two[:int(edge_weight_list_union_len/2)], edge_grads_two[int(edge_weight_list_union_len/2):]

                # source_to_target
                h_index = torch.argmax(x_source_vec,dim=0)
                t_index = torch.argmax(x_target_vec,dim=0)
                distances, back_edges = beam_search_distance(edge_index, edge_type, num_of_nodes, edge_grads_1, h_index, t_index, num_beam=config['num_beam'])
                paths_1, weights_1 = topk_average_length(distances, back_edges, t_index, config['path_topk'])
                paths_org_1 = []
                for path_idx in range(len(weights_1)):
                    weight = weights_1[path_idx]
                    path = paths_1[path_idx]
                    path_org = []
                    for path_edge in path:
                        path_org.append((id[path_edge[0]], id[path_edge[1]], path_edge[2]))
                    paths_org_1.append(path_org)
                print('paths_1: ', paths_1)
                print('paths_org_1:', paths_org_1)
                print('weights_1', weights_1)

                # target_to_source
                h_index = torch.argmax(x_target_vec,dim=0)
                t_index = torch.argmax(x_source_vec,dim=0)
                distances, back_edges = beam_search_distance(edge_index[[1,0],:], edge_type, num_of_nodes, edge_grads_2, h_index, t_index, num_beam=config['num_beam'])
                paths_2, weights_2 = topk_average_length(distances, back_edges, t_index, config['path_topk'])
                paths_org_2 = []
                for path_idx in range(len(weights_2)):
                    weight = weights_2[path_idx]
                    path = paths_2[path_idx]
                    path_org = []
                    for path_edge in path:
                        path_org.append((id[path_edge[0]], id[path_edge[1]], path_edge[2]))
                    paths_org_2.append(path_org)
                print('paths_2: ', paths_2)
                print('paths_org_2:', paths_org_2)
                print('weights_2', weights_2)

                paths_str_1 = join_path(paths_org_1)
                weights_str_1 = ';'.join([str(weight) for weight in weights_1])
                paths_str_2 = join_path(paths_org_2)
                weights_str_2 = ';'.join([str(weight) for weight in weights_2])

                assert (len(ids) == len(labels) == len(predictions))
                values = ["\t".join([str(qid), str(label), str(p), str(emb), str(paths_1), str(weights_1), str(paths_2), str(weights_2)]) for qid, label, p, emb, paths_1, weights_1, paths_2, weights_2 in zip(ids, labels, predictions, out_embeddings, [paths_str_1,],[weights_str_1,],[paths_str_2,],[weights_str_2,])]
                writer.write("\n".join(values)+"\n")
            else:
                assert (len(ids) == len(labels) == len(predictions))
                if bool_flag(config['out_emb']):
                    values = ["\t".join([str(qid), str(label), str(emb)]) for qid, label, emb in zip(ids, labels, out_embeddings)]
                    writer.write("\n".join(values) + "\n")
                else:
                    values = ["\t".join([str(qid), str(label), str(p)]) for qid, label, p in zip(ids, labels, predictions)]
                    writer.write("\n".join(values) + "\n")

            if (step) % config['num_step_print'] == 0:
                printWithTimes('Step [%d/%d]' % (step, num_step_per_epoch))

            step += 1

    writer.close()

hete_graph.g.close()



