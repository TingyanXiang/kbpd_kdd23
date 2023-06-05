# coding=utf-8
import graphlearn as gl
import numpy as np
import graphlearn.python.nn.pytorch as thg
import os

class BaseGraph(object):
    def __init__(self, params, conf=None):
        self.params = params
        self.conf = conf
        self.g = self._init_graph()

    def _init_graph(self):
        raise NotImplementedError('_init_graph not implemented')

    def feed_next(self, phase, batch_size):
        raise NotImplementedError('feed_next not implemented')

    def init_gl(self, rank, world_size):
        '''
        init graph object
        '''
        self.g.init(task_index=rank, task_count=world_size, hosts=thg.get_cluster_spec()['server'])

class HeteGraph(BaseGraph):
    def __init__(self, params, conf=None):
        super(HeteGraph, self).__init__(params, conf)

    def _init_graph(self):
        g = gl.Graph()

        print(self.params)

        root_path = self.params['data_root_path']
        tables_dict = self.params['tables_dict']
        nodetables_dict = tables_dict.get('node',dict())
        edgetables_dict = tables_dict.get('edge',dict())
        sampletables_dict = tables_dict.get('sample',dict())

        print('node')
        for table_name, config in nodetables_dict.items():
            print(table_name, config)
            g.node(source=os.path.join(root_path,table_name), node_type=config['node_type'],
                   decoder=config['decoder'])
        print('edge')
        for table_name, config in edgetables_dict.items():
            print(table_name, config)
            g.edge(source=os.path.join(root_path,table_name), edge_type=config['edge_type'],
                   decoder=config['decoder'], directed=config.get('directed',True))
        print('sample')
        for table_name, config in sampletables_dict.items():
            print(table_name, config)
            mask = config.get('mask', None)
            if mask is not None:
                g.node(source=os.path.join(root_path,table_name), node_type=config['node_type'], decoder=config['decoder'], mask=mask)
            else:
                g.node(source=os.path.join(root_path,table_name), node_type=config['node_type'], decoder=config['decoder'])

        return g

    def feed_next_nodeonly(self, batch_size, node_type, mask=gl.Mask.TRAIN):
        query = self.g.V(node_type, mask=mask).batch(batch_size).shuffle(traverse=True).alias('src').values()
        return query

    def feed_next_w1hop(self, batch_size, node_type, mask=gl.Mask.TRAIN, edge_types_config=dict()):
        seed = self.g.V(node_type, mask=mask).batch(batch_size).shuffle(traverse=True).alias('src')
        for edge_type, config in edge_types_config.items():
            seed.outV(edge_type).sample(config['nbrs_num']).by('random').alias('src_hop1_'+edge_type)
        return seed.values()

    def get_stats(self):
        return self.g.get_stats()
