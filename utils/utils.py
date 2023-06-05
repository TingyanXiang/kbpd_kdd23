import json
import os
import time
import datetime
import torch

def save_model(model, model_path, model_name, save_mode='state_dict'):
    final_path = os.path.join(model_path, model_name)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    if save_mode == 'state_dict':
        torch.save(model.state_dict(), os.path.join(final_path,model_name+'.pt'))
    else:
        torch.save(model, os.path.join(final_path,model_name+'.pt'))
    return None

def load_model(model, model_path, model_name, save_mode='state_dict'):
    final_path = os.path.join(model_path, model_name)
    if not os.path.exists(final_path):
        raise IOError("{final_path} does not existed!!!".format(final_path=final_path))
    if save_mode == 'state_dict':
        model.load_state_dict(torch.load(os.path.join(final_path,model_name+'.pt')))
    else:
        model = torch.load(final_path)
    return model

def printWithTimes(line):
    print('%s %s' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), line))

class Timer(object):
    def __init__(self, times=100, print_prefix='Timer', close=False):

        self.start_time_dict = {}
        self.end_time_dict = {}
        self.times = times
        self.print_prefix = print_prefix
        self.close = close

    def start(self, key):
        if self.close: return None
        if key not in self.start_time_dict.keys():
            self.start_time_dict[key] = []

        self.start_time_dict[key].append(time.time())

    def end(self, key):
        if self.close: return None
        if key not in self.end_time_dict.keys():
            self.end_time_dict[key] = []

        self.end_time_dict[key].append(time.time())

        assert len(self.start_time_dict[key]) == len(
            self.end_time_dict[key]), 'start_time_dict[key]:%s end_time_dict[key]:%s' % (
        str(self.start_time_dict[key]), str(self.end_time_dict[key]))

        if len(self.end_time_dict[key]) == self.times:

            total_time = 0
            for start_time, end_time in zip(self.start_time_dict[key], self.end_time_dict[key]):
                time_ = end_time - start_time  # 注意，不可以命名成 time，会覆盖掉time这module
                total_time += time_

            avg_time = total_time / self.times

            print("【%s】"%(self.print_prefix), key, 'average time:', '%.6f' % (avg_time))

            # 清空
            self.start_time_dict[key] = []
            self.end_time_dict[key] = []

    def reset(self, key=None):
        if key is None:
            self.start_time_dict = {}
            self.end_time_dict = {}
        else:
            self.start_time_dict[key] = []
            self.end_time_dict[key] = []

    def print(self):
        print('start_time_dict:%s end_time_dict:%s' % (str(self.start_time_dict), str(self.end_time_dict)))

def bool_flag(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')

def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)
