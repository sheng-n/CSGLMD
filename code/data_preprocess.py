from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import torch
import scipy.sparse as sp

"TThis code uses one-fold cross-validation as an example"

class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.label = triple[:, 2]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        return self.label[index], (self.entity1[index], self.entity2[index])


def load_data(args, test_ratio=0.2):
    """Read data from path, convert data into loader, return features and adjacency"""
    # read data
    print('Loading {0} seed{1} dataset...'.format(args.in_file, args.seed))  #
    positive = np.loadtxt(args.in_file, dtype=np.int64)

    # postive sample
    link_size = int(positive.shape[0])
    np.random.seed(args.seed)
    np.random.shuffle(positive)
    positive = positive[:link_size]

    # negative sample
    negative_all = np.loadtxt(args.neg_sample, dtype=np.int64)
    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:positive.shape[0]])

    test_size = int(test_ratio * positive.shape[0])
    # print("positive_data:",val_size,test_size)

    positive = np.concatenate([positive, np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)], axis=1)
    negative = np.concatenate([negative, np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)], axis=1)
    negative_all = np.concatenate([negative_all, np.zeros(negative_all.shape[0], dtype=np.int64).reshape(negative_all.shape[0], 1)], axis=1) #94513,3

    print('Selected cross_validation type...')
    if args.validation_type == '5_cv1':
        train_data = np.vstack((positive[: -test_size], negative[: -test_size]))
        test_data = np.vstack((positive[-test_size:], negative[-test_size:]))
        print("data: ",train_data.shape,test_data.shape)

    elif args.validation_type == '5_cv2':
        train_data = np.vstack((positive[: -test_size], negative[: -test_size]))
        test_data = np.vstack((positive[-test_size:], negative_all))
        print("data: ",train_data.shape,test_data.shape)


    # construct adjacency
    train_positive = positive[: -test_size]
    # print("train_positive: ",train_positive)
    print('Selected task type...')
    "Note: node similarity need to recomputed, (1)your can save train/test id, " \
    "(2) according to calculating_similarity.py. "

    if args.task_type == 'LDA':
        l_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                            shape=(240, 405), dtype=np.float32)  # dataset1 shape=(240,405) dataset2 shape=(665, 316)
        l_d = l_d.toarray()  # 240, 405
        m_d = np.loadtxt("dataset1/mi_dis.txt")   # 495, 405
        m_l = np.loadtxt("dataset1/lnc_mi.txt").T  # 495, 240
        l_sim = np.loadtxt("dataset1/one_hot_lnc_sim.txt") # 240,240
        d_sim = np.loadtxt("dataset1/one_hot_dis_sim.txt")  # 405,405
        m_sim = np.loadtxt("dataset1/one_hot_mi_sim.txt")   # 495,495

    if args.task_type == 'MDA':
        m_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                            shape=(495, 405), dtype=np.float32)  # dataset1 shape=(495,405) dataset2 shape=(295, 316)
        m_d = m_d.toarray()  #495,405
        l_d = np.loadtxt("dataset1/lnc_dis.txt")  #495, 405
        m_l = np.loadtxt("dataset1/lnc_mi.txt").T  # 495, 240

        l_sim = np.loadtxt("dataset1/lnc_fuse_sim_0.8.txt")
        d_sim = np.loadtxt("dataset1/dis_fuse_sim_0.8.txt")
        m_sim = np.loadtxt("dataset1/mi_fuse_sim_0.8.txt")

    if args.task_type == 'LMI':
        l_m = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                            shape=(240, 495), dtype=np.float32)  # dataset1 shape=(240,495) dataset2 shape=(665, 295)
        m_l = l_m.toarray().T
        l_d = np.loadtxt("dataset1/lnc_dis.txt")  # 495, 405
        m_d = np.loadtxt("dataset1/mi_dis.txt")   # 495, 405
        l_sim = np.loadtxt("dataset1/one_hot_lnc_sim.txt") # 240,240
        d_sim = np.loadtxt("dataset1/one_hot_dis_sim.txt")  # 405,405
        m_sim = np.loadtxt("dataset1/one_hot_mi_sim.txt")  # 495,495
    # print(l_d.shape, m_d.shape, l_m.shape, l_sim.shape, d_sim.shape, m_sim.shape)

    adj = construct_graph(l_d, m_d, m_l, l_sim, m_sim, d_sim)
    adj = lalacians_norm(adj)

    # construct edges
    edges_o = adj.nonzero()
    edge_index_o = torch.tensor(np.vstack((edges_o[0], edges_o[1])), dtype=torch.long)

    # build data loader
    params = {'batch_size': args.batch, 'shuffle': True,  'drop_last': True}

    training_set = Data_class(train_data)
    train_loader = DataLoader(training_set, **params)

    test_set = Data_class(test_data)
    test_loader = DataLoader(test_set, **params)

    # extract features
    print('Extracting features...')
    if args.feature_type == 'one_hot':
        features = np.eye(adj.shape[0])

    elif args.feature_type == 'uniform':
        np.random.seed(args.seed)
        features = np.random.uniform(low=0, high=1, size=(adj.shape[0], args.dimensions))

    elif args.feature_type == 'normal':
        np.random.seed(args.seed)
        features = np.random.normal(loc=0, scale=1, size=(adj.shape[0], args.dimensions))

    elif args.feature_type == 'position':
        features = sp.coo_matrix(adj).todense()

    features_o = normalize(features)
    args.dimensions = features_o.shape[1]

    # adversarial nodes
    np.random.seed(args.seed)
    id = np.arange(features_o.shape[0])
    id = np.random.permutation(id)
    features_a = features_o[id]

    y_a = torch.cat((torch.ones(adj.shape[0], 1), torch.zeros(adj.shape[0], 1)), dim=1)

    x_o = torch.tensor(features_o, dtype=torch.float)
    data_o = Data(x = x_o, edge_index=edge_index_o)
    # print(data_o)

    x_a = torch.tensor(features_a, dtype=torch.float)
    data_a = Data(x=x_a, y=y_a)
    # print(data_a)

    print('Loading finished!')
    return data_o, data_a, train_loader, test_loader
