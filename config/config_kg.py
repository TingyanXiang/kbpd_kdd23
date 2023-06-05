import graphlearn as gl

config = {
    #run process
    'model_name': 'nbfnet_test2',
    'ddp': False, #torch distributed data parallel
    'mode': None, #'train'/'test' could read from commandline
    'num_epoch': 2,
    'batch_size': 4,
    # 'case_study': 'false',
    'out_emb': 'True',
    'save_step_cnt': 100,
    'num_step_print': 1, #print frequency about model running info
    'save_root_path': None, # where to save model and outputs, default='./saved_models/'

    #optimizer
    'learning_rate': 0.001,

    # model
    'model_choice': 'nbfnet',
    'ablation': [], #[exppath_sep,]: output path importance
    'flow': 'bi_direction',
    'short_cut': 'false',
    'node_feat_num': 10,
    'workgraph_feat_num': 20*6+1,
    'workgraph_edgestr_len': 6,
    'worknode_feat_num': 10,
    'workrelation_feat_num': 10,
    'num_class': 2,
    'gnn_num_layers': 1,
    'mlp_num_layers': 2,
    'message_func': 'add',
    'aggregate_func': 'max',
    'cate_choice': 'lastgnn_qa', #lastgnn_query_qa/lastgnn_qa
    'edge_model_share': 'false',
    'node_model_share': 'false',
    'layer_norm': 'false',
    'batch_norm': 'false',
    'gnn_hidden_dim': 3, #hidden_dim in nfbnet gnn modules
    'dropout_rate': 0.5,
    
    # eval
    'num_beam': 20,
    'path_topk': 20,
}

# graph config
graph_config= {
    'data_root_path': None #default='./kbpd_kdd23_open_code_v2/data/'
    , 'tables_dict': {
        # node info in graph
        'node': {
            # key = file_name; value = setting in graphlearn
            # kg_sample_node_table attrs includes 2 parts, int_attrs and float_attrs.
            ## int_attrs: the subgraph retrieved from RDKG with format = (spo_cnt, s1, s1_type, p1, p1_weight, o1, o1_type, ..., sn, sn_type, pn, pn_weight, on, on_type), and p_weight is optional.
            ## float_attrs: context embeddings of <item, risk-point>
            'kg_sample_node_table': {'node_type': 'samplenode', 'decoder': gl.Decoder(labeled=True,
                                                                                      attr_types=['int'] * config['workgraph_feat_num'] + ['float'] * config['node_feat_num'],
                                                                                      attr_delimiter=',')}
            , 'kg_work_node_table': {'node_type': 'worknode', 'decoder': gl.Decoder(labeled=True,
                                                                                    attr_types=['float'] * config['worknode_feat_num'],
                                                                                    attr_delimiter=',')}
            , 'kg_work_relation_table': {'node_type': 'workrelation',
                                         'decoder': gl.Decoder(attr_types=['float'] * config['workrelation_feat_num'],
                                                               attr_delimiter=',')}
        }
        # edge info in graph
        , 'edge': {}
        # sample table: partial data from node_type node table
        , 'sample': {
            # key = file_name; value = setting in graphlearn
            'kg_sample_table': {'node_type': 'samplenode', 'mask': gl.Mask.TRAIN,
                                'decoder': gl.Decoder(labeled=True, attr_delimiter=',')}
        }
    }
    ,'source_node_type': 0 #in kg subgraph, source_node_type; by default, source=risk point;
    ,'target_node_type': 2 #in kg subgraph, target_node_type; by default, target=item;
}