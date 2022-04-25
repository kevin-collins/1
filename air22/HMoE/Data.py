import csv
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import joblib
import re
import random
import time
from options import args_parser
from utils import exp_details, cal_loss, cal_auc, test, get_model, get_user_groups
"""
sample_skeleton_train.csv
# id   点击 转换  common_features索引  特征数量  特征
['1', '0', '0', 'bacff91692951881' , '9', xxxxx

common_features_train.csv
# common_features索引  特征数量  特征
'bacff91692951881'  xxx  xxx

其中特征是以  feature_field_id 0x02 feature_id 0x03 feature_value  组成的

如 \x01101\x0231319\x031.0  表示101类型的特征  特征id为31319  值为1.0
"""


'''
process the Ali-CCP (Alibaba Click and Conversion Prediction) dataset.
https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408

@The author:
Dongbo Xi (xidongbo@meituan.com)
'''

data_path = '/home/clq/datas/ali-ccp/sample_skeleton_{}.csv'
common_feat_path = '/home/clq/datas/ali-ccp/common_features_{}.csv'
enum_path = '/home/clq/datas/ali-ccp/ctrcvr_enum.pkl'
write_path = '/home/clq/datas/ali-ccp/ctr_cvr'
vocabulary_size = {
    '101': 238635,
    '121': 98,
    '122': 14,
    '124': 3,
    '125': 8,
    '126': 4,
    '127': 4,
    '128': 3,
    '129': 5,
    '205': 467298,
    '206': 6929,
    '207': 263942,
    '216': 106399,
    '508': 5888,
    '509': 104830,
    '702': 51878,
    '853': 37148,
    '301': 4
}

use_columns = [
    '101',
    '121',
    '122',
    '124',
    '125',
    '126',
    '127',
    '128',
    '129',
    '205',
    '206',
    '207',
    '216',
    '508',
    '509',
    '702',
    '853',
    '301']

class process(object):
    def __init__(self):
        pass

    def process_train(self):
        c = 0
        common_feat_dict = {}
        # 得到分割好的公共特征
        with open(common_feat_path.format('train'), 'r') as fr:
            for line in fr:
                line_list = line.strip().split(',')
                # 分割特征
                kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
                # 获得特征id
                key = kv[range(0, len(kv), 3)]
                # \x01后面的是值
                value = kv[range(1, len(kv), 3)]
                feat_dict = dict(zip(key, value))  # key和value的字典
                common_feat_dict[line_list[0]] = feat_dict  # id和特征的字典
                c += 1
                if c % 100000 == 0:
                    print(c)
        print('join feats...')
        c = 0
        # 记录某个特征下 各个值出现的次数
        vocabulary = dict(zip(use_columns, [{} for _ in range(len(use_columns))]))
        # 得到分割好的骨架特征
        with open(data_path.format('train') + '.tmp', 'w') as fw:
            fw.write('click,purchase,' + ','.join(use_columns) + '\n')
            with open(data_path.format('train'), 'r') as fr:
                for line in fr:
                    line_list = line.strip().split(',')
                    if line_list[1] == '0' and line_list[2] == '1':
                        continue
                    kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
                    key = kv[range(0, len(kv), 3)]
                    value = kv[range(1, len(kv), 3)]
                    feat_dict = dict(zip(key, value))
                    # 插入公共特征
                    feat_dict.update(common_feat_dict[line_list[3]])
                    # feats是特征列表 没有的特征值为0
                    feats = line_list[1:3]
                    for k in use_columns:
                        feats.append(feat_dict.get(k, '0'))
                    # 写入文件
                    fw.write(','.join(feats) + '\n')
                    # 计算各个特征值出现的次数
                    for k, v in feat_dict.items():
                        if k in use_columns:
                            if v in vocabulary[k]:
                                vocabulary[k][v] += 1
                            else:
                                vocabulary[k][v] = 0
                    c += 1
                    if c % 100000 == 0:
                        print(c)
        print('before filter low freq:')
        # 每个特征对应多少可能的值
        for k, v in vocabulary.items():
            print(k + ':' + str(len(v)))
        new_vocabulary = dict(
            zip(use_columns, [set() for _ in range(len(use_columns))]))
        for k, v in vocabulary.items():
            for k1, v1 in v.items():  # k1是特征值  v1是特征值出现的次数
                if v1 > 10:  # 超过10次才记录
                    new_vocabulary[k].add(k1)
        # 字典 记录特征名 和 出现超过10次特征值
        vocabulary = new_vocabulary
        print('after filter low freq:')
        for k, v in vocabulary.items():
            print(k + ':' + str(len(v)))
        joblib.dump(vocabulary, enum_path, compress=3)

        print('encode feats...')
        vocabulary = joblib.load(enum_path)
        feat_map = {}
        # 把值进行数字化 比如原来是35，36 变成 1，2
        for feat in use_columns:
            feat_map[feat] = dict(
                zip(vocabulary[feat], range(1, len(vocabulary[feat]) + 1)))
        c = 0
        # 写入特征训练文件
        with open(write_path + '.train', 'w') as fw1:
            with open(write_path + '.dev', 'w') as fw2:
                fw1.write('click,purchase,' + ','.join(use_columns) + '\n')
                fw2.write('click,purchase,' + ','.join(use_columns) + '\n')
                with open(data_path.format('train') + '.tmp', 'r') as fr:
                    fr.readline()  # remove header

                    for line in fr:
                        line_list = line.strip().split(',')
                        new_line = line_list[:2]
                        for value, feat in zip(line_list[2:], use_columns):
                            new_line.append(
                                str(feat_map[feat].get(value, '0')))
                        # 1：9 随机划分
                        """
                        if random.random() >= 0.9:
                            fw2.write(','.join(new_line) + '\n')
                        else:
                            fw1.write(','.join(new_line) + '\n')
                        """
                        # 按顺序
                        if c < 38070370:
                            fw1.write(','.join(new_line) + '\n')
                        else:
                            fw2.write(','.join(new_line) + '\n')
                        c += 1
                        if c % 100000 == 0:
                            print(c)

    def process_test(self):
        c = 0
        common_feat_dict = {}
        with open(common_feat_path.format('test'), 'r') as fr:
            for line in fr:
                line_list = line.strip().split(',')
                kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
                key = kv[range(0, len(kv), 3)]
                value = kv[range(1, len(kv), 3)]
                feat_dict = dict(zip(key, value))
                common_feat_dict[line_list[0]] = feat_dict
                c += 1
                if c % 100000 == 0:
                    print(c)
        print('join feats...')
        c = 0
        with open(data_path.format('test') + '.tmp', 'w') as fw:
            fw.write('click,purchase,' + ','.join(use_columns) + '\n')
            with open(data_path.format('test'), 'r') as fr:
                for line in fr:
                    line_list = line.strip().split(',')
                    if line_list[1] == '0' and line_list[2] == '1':
                        continue
                    kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
                    key = kv[range(0, len(kv), 3)]
                    value = kv[range(1, len(kv), 3)]
                    feat_dict = dict(zip(key, value))
                    feat_dict.update(common_feat_dict[line_list[3]])
                    feats = line_list[1:3]
                    for k in use_columns:
                        feats.append(str(feat_dict.get(k, '0')))
                    fw.write(','.join(feats) + '\n')
                    c += 1
                    if c % 100000 == 0:
                        print(c)

        print('encode feats...')
        vocabulary = joblib.load(enum_path)
        feat_map = {}
        for feat in use_columns:
            feat_map[feat] = dict(
                zip(vocabulary[feat], range(1, len(vocabulary[feat]) + 1)))
        c = 0
        with open(write_path + '.test', 'w') as fw:
            fw.write('click,purchase,' + ','.join(use_columns) + '\n')
            with open(data_path.format('test') + '.tmp', 'r') as fr:
                fr.readline()  # remove header
                for line in fr:
                    line_list = line.strip().split(',')
                    new_line = line_list[:2]
                    for value, feat in zip(line_list[2:], use_columns):
                        new_line.append(str(feat_map[feat].get(value, '0')))
                    fw.write(','.join(new_line) + '\n')
                    c += 1
                    if c % 100000 == 0:
                        print(c)


class XDataset(Dataset):
    '''load csv data with feature name ad first row'''
    def __init__(self, datafile):
        super(XDataset, self).__init__()
        self.feature_names = []
        self.datafile = datafile
        self.data = []
        self._load_data()

    def _load_data(self):
        start_time = time.time()
        print("start load data from: {}".format(self.datafile))
        count = 0
        with open(self.datafile) as f:
            self.feature_names = f.readline().strip().split(',')[2:]
            for line in f:
                count += 1
                line = line.strip().split(',')
                line = [int(v) for v in line]
                self.data.append(line)
        print("load data from {} finished".format(self.datafile))
        print("nums %.3f  cost time %.3f s" % (count, time.time()-start_time))

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        click = line[0]
        conversion = line[1]
        features = dict(zip(self.feature_names, line[2:]))
        return click, conversion, features


def get_dataloader(filename, batch_size, shuffle):
    data = XDataset(filename)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_dataset(filename):
    data = XDataset(filename)
    return data

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


def save_feature_by_address(model_path="/home/clq/projects/HMoE/main_models/2022-04-05 14:02:14/main_best.pth"):
    args = args_parser()
    args.scenario_nums = 3  # defaultdict(<class 'int'>, {'2': 14296532, '3': 286913, '1': 23487225})
    args.input_size = len(vocabulary_size) * args.embedding_size
    exp_details(args)  # 输出参数

    model = get_model(args, 'HMoE', vocabulary_size)
    model.load_state_dict(torch.load(model_path))

    train_dataset = get_dataset(args.path_train)
    train_idxs, dev_idxs, test_idxs, args.num_clients, args.scenario_num = get_user_groups(train_dataset, None, None)

    datas = []

    # 对每个地区进行分群
    for i in range(args.num_clients):
        train_local_dataset = DatasetSplit(train_dataset, train_idxs[i])
        group = dict()
        embeddings = []
        for i, (click, conversion, features) in enumerate(train_local_dataset):
            for key in features.keys():
                features[key] = features[key].to(args.device)
            embeddings.append(model(features))
            for key in group.keys():




"""
class Data(object):
    def __init__(self, args):
        self.path_skeleton = args.path_skeleton
        self.path_common = args.path_common
        datas = self.GetData(args)

    def GetData(self, args):
        csv_reader_skeleton = csv.reader(open(self.path_skeleton))
        csv_reader_common = csv.reader(open(self.path_common))
        skeleton = []
        common = dict()
        combined = []
        for line in csv_reader_skeleton:
            skeleton.append(line)
        for line in csv_reader_common:
            common[line[0]] = (line[1], line[2])
        for line in skeleton:
            common_id = line[3]
            common_data = common[common_id]
            feature_nums = int(line[4]) + int(common_data[1])
            feature = line[-1] + common_data[-1]
            combined.append((line[0], line[1], line[2], feature_nums, feature))
        return combined




address = defaultdict(int)
csv_reader_skeleton = csv.reader(open("E:/datas/common_features_train.csv"))
for line in csv_reader_skeleton:
    features = line[-1].split('\x01')
    for f in features:
        name = f.split('\x02')[0]
        if name == '129':
            # print(f.split('\x02')[1].split('\x03')[0])
            address[f.split('\x02')[1].split('\x03')[0]] += 1
print(address)

# defaultdict(<class 'int'>, {'3864888': 205365, '3864889': 111358, '3864890': 87730, '3864887': 51070})


if __name__ == "__main__":
    pros = process()
    pros.process_train()
    # pros.process_test()


['810d5366057b3f58', '1025', '101\x02412797\x031.0       \x01 125 \x02 3438772 \x03 1.0

csv_reader_skeleton = csv.reader(open("/home/clq/datas/ali-ccp/sample_skeleton_test.csv"))
for line in csv_reader_skeleton:
    print(line)
    features = line[-1].split('\x01')
    for f in features:
        name = f.split('\x02')[0]
        if name == '125':
            print(f.split('\x02')[1].split('\x03')[0])
    exit(0)

datas = [
    dict(
    zip(use_columns, [{'mean':0, 'var':0, 'n':0} for _ in range(len(use_columns))]))
    for _ in range(5)
]
with open("/home/clq/datas/ali-ccp/ctr_cvr.train") as f:
    line = f.readline().split(',')
    line[-1] = line[-1][:-1]
    head = {i: key for i,key in enumerate(line)}
    for line in f:
        data = line.split(',')
        add = int(data[10])
        for i,v in enumerate(data):
            if i!=10 and i!=0 and i!=1:
                value = int(data[i])
                incre_avg = (datas[add][head[i]]['n']*datas[add][head[i]]['mean']+value)/(datas[add][head[i]]['n']+1)
                incre_std = np.sqrt((datas[add][head[i]]['n']*(datas[add][head[i]]['var']**2+(incre_avg-datas[add][head[i]]['mean'])**2)+(incre_avg-value)**2)/(datas[add][head[i]]['n']+1))
                datas[add][head[i]]['mean'] = incre_avg
                datas[add][head[i]]['var'] = incre_std
                datas[add][head[i]]['n'] += 1
print(datas)
for i,x in enumerate(datas):
    print(i)
    for k,v in x.items():
        print(k,v)
        

if __name__ == "__main__":
    pros = process()
    pros.process_train()
    # pros.process_test()

n = [0.0,0.0,0.0,0.0,0.0]
datas = [
    dict(
    zip(use_columns, [0 for _ in range(len(use_columns))]))
    for _ in range(5)
]
with open("/home/clq/datas/ali-ccp/ctr_cvr.train") as f:
    line = f.readline().split(',')
    line[-1] = line[-1][:-1]
    head = {i: key for i,key in enumerate(line)}
    for line in f:
        data = line.split(',')
        add = int(data[10])
        n[add] += 1
        for i, v in enumerate(data):
            if i != 10 and i!=0 and i!=1:
                value = int(data[i])
                if value == 0:
                    datas[add][head[i]] += 1

print(datas)
for i, x in enumerate(datas):
    print(i)
    for k, v in x.items():
        print(k, v, v/n[i])
        
"""