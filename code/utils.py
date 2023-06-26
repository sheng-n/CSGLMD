from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp

def normalize(mx):
    '''Row-normalize sparse matrix'''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def Preproces_Data (A, test_id):
    copy_A = A / 1
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A

'''construct_graph (lncRNA, disease, miRNA)'''
def construct_graph(lncRNA_disease,  miRNA_disease, miRNA_lncRNA, lncRNA_sim, miRNA_sim, disease_sim ):
    lnc_dis_sim = np.hstack((lncRNA_sim, lncRNA_disease, miRNA_lncRNA.T))
    dis_lnc_sim = np.hstack((lncRNA_disease.T, disease_sim, miRNA_disease.T))
    mi_lnc_dis = np.hstack((miRNA_lncRNA, miRNA_disease, miRNA_sim))

    matrix_A = np.vstack((lnc_dis_sim, dis_lnc_sim, mi_lnc_dis))          # dataset1 1140,1140
    return matrix_A

'''Norm'''
def lalacians_norm(adj):
    # adj += np.eye(adj.shape[0]) # add self-loop
    degree = np.array(adj.sum(1))
    D = []
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))
    norm_A = degree.dot(adj).dot(degree)
    # norm_A = degree.dot(adj)
    return norm_A