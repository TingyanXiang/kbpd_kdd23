import graphlearn as gl

config = {
    # run process
    'model_name': 'rgcn_test0',
    'ddp': False,  # torch distributed data parallel
    'mode': None, #'train'/'test'; could read from commandline
    'num_epoch': 2,
    'batch_size': 4,
    'out_emb': 'false', #True/False: output kg embedding for hrg fusion
    #'save_step_cnt': 100,
    'num_step_print': 1, #print frequency about model running info
    'save_root_path': None, # where to save model and outputs, default='./saved_models/'

    # optimizer
    'learning_rate': 0.001,

    # model
    'model_choice': 'RGCN',
    'nbrs_num': 5,
    'ablation': [],
    'node_feat_num': 13, #input_dim
    'gnn_num_layers': 1,
    'dropout_rate': 0.5,
    # 'in_channels': 8, #==node_feat_num
    'hidden_channels': 8,
    'num_class': 2,
    'num_relations': 3,
    'n_bases': 6,
    'out_layer_num': 1
}

# graph config
graph_config = {
    'data_root_path': None #default='./kbpd_kdd23/data/'
    , 'meta_data': [['samplenode'],[('samplenode','sameseller','samplenode'),('samplenode','relatedseller','samplenode'),('samplenode','visit','samplenode')]]
    , 'default_label': -1 #default_label for nodes without labels
    , 'default_neighbor_id': 0  #default_node_id is used if there is no enough neibors when samping
    , 'tables_dict': {
        # node info in graph
        'node': {
            # key = file_name; value = setting in graphlearn
            'hrg_sample_node_table': {'node_type': 'samplenode', 'decoder': gl.Decoder(labeled=True,
                                                                                      attr_types=['float'] * config['node_feat_num'],
                                                                                      attr_delimiter=',')}
        }
        # edge info in graph
        , 'edge': {
            # key = file_name; value = setting in graphlearn
            'hrg_edge_table_sameseller': {'edge_type': ('samplenode','samplenode','sameseller'), 'decoder': gl.Decoder(), 'directed': False}
            ,'hrg_edge_table_relatedseller': {'edge_type': ('samplenode','samplenode','relatedseller'), 'decoder': gl.Decoder(), 'directed': False}
            , 'hrg_edge_table_visit': {'edge_type': ('samplenode','samplenode','visit'), 'decoder': gl.Decoder(), 'directed': False}
        }
        # sample table: partial data from node_type node table
        , 'sample': {
            # key = file_name; value = setting in graphlearn
            'hrg_sample_table': {'node_type': 'samplenode', 'mask': gl.Mask.TRAIN,
                                'decoder': gl.Decoder(labeled=True, attr_delimiter=',')}
        }
    }
}