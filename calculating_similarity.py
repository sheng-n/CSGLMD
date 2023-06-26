import numpy as np
import copy

"positive sample in test set to 0"
def Preproces_Data(A, test_id):
    copy_A = A / 1
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A

"Gaussiankernel similarity"
def calculate_kernel_bandwidth(A):
    IP_0 = 0
    for i in range(A.shape[0]):
        IP = np.square(np.linalg.norm(A[i]))
        # print(IP)
        IP_0 += IP
    lambd = 1/((1/A.shape[0]) * IP_0)
    return lambd

def calculate_GaussianKernel_sim(A):
    kernel_bandwidth = calculate_kernel_bandwidth(A)
    gauss_kernel_sim = np.zeros((A.shape[0],A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))
            gauss_kernel_sim[i][j] = gaussianKernel
            # print("gau",gauss_kernel_sim)

    return gauss_kernel_sim

"Functional similarity"
def PBPA(RNA_i, RNA_j, di_sim, rna_di):
    diseaseSet_i = rna_di[RNA_i] > 0

    diseaseSet_j = rna_di[RNA_j] > 0
    diseaseSim_ij = di_sim[diseaseSet_i][:, diseaseSet_j]
    ijshape = diseaseSim_ij.shape
    if ijshape[0] == 0 or ijshape[1] == 0:
        return 0
    return (sum(np.max(diseaseSim_ij, axis=0)) + sum(np.max(diseaseSim_ij, axis=1))) / (ijshape[0] + ijshape[1])

def getRNA_functional_sim(RNAlen, diSiNet, rna_di):
    RNASiNet = np.zeros((RNAlen, RNAlen))
    for i in range(RNAlen):
        for j in range(i + 1, RNAlen):
            RNASiNet[i, j] = RNASiNet[j, i] = PBPA(i, j, diSiNet, rna_di)
    RNASiNet = RNASiNet + np.eye(RNAlen)
    return RNASiNet

"label instantiation"
def label_preprocess(sim_matrix):
    new_sim_matrix = np.zeros(shape=sim_matrix.shape)
    # print(lnc_sim_matrix.shape)
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if sim_matrix[i][j] >= 0.8:
                new_sim_matrix[i][j] = 1

    return new_sim_matrix


def RNA_fusion_sim (G1, G2, F):
    fusion_sim = np.zeros((len(G1),len(G2)))
    G = (G1+G2)/2
    for i in range (len(G1)):
        for j in range(len(G1)):
            if F[i][j] > 0 :
                fusion_sim[i][j] = F[i][j]
            else:
                fusion_sim[i][j] = G[i][j]
    fusion_sim = label_preprocess(fusion_sim)
    return fusion_sim

def dis_fusion_sim (G1, G2, SD):
    fusion_sim = (SD+(G1+G2)/2)/2
    fusion_sim = label_preprocess(fusion_sim)
    return fusion_sim


if __name__ == '__main__':

    'dataset1'
    lnc_dis = np.loadtxt("dataset1/lnc_dis_association.txt")  # 240,405,2687
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")  # 495,405,13559
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")  # 240,495,1002
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")  # 405,405
    print(lnc_dis.shape,mi_dis.shape,lnc_mi.shape,dis_sem_sim.shape)

    'dataset2'
    lnc_dis = np.loadtxt("dataset1/lnc_dis.txt")  # 665,316
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")  # 295,316
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")  # 665,295
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")  # 316,316
    print(lnc_dis.shape,mi_dis.shape,lnc_mi.shape,dis_sem_sim.shape)

    "this example use all sample to calculate"

    # lnc_dis_test_id = np.loadtxt("dataset1/lnc_dis_test_id1.txt")  # 4289,2
    # mi_dis_test_id = np.loadtxt("dataset1/mi_dis_test_id1.txt")  # 21694,2
    # mi_lnc_test_id = np.loadtxt("dataset1/mi_lnc_test_id1.txt")  # 1602,2
    #
    # "Zeroing of the association matrix"
    # lnc_dis = Preproces_Data(lnc_dis,lnc_dis_test_id)
    # mi_dis = Preproces_Data(mi_dis,mi_dis_test_id)
    # mi_lnc = Preproces_Data(lnc_mi.T,mi_lnc_test_id)   #495,240
    # # print(mi_lnc.shape)

    "lncRNA similarity"
    lnc_gau_1 = calculate_GaussianKernel_sim(lnc_dis)  # lncRNA-disease
    lnc_gau_2 = calculate_GaussianKernel_sim(lnc_mi)   # lncRNA-miRNA
    lnc_fun = getRNA_functional_sim(RNAlen=len(lnc_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(lnc_dis))
    lnc_sim = RNA_fusion_sim(lnc_gau_1,lnc_gau_2,lnc_fun)

    "miRNA similarity"
    mi_gau_1 = calculate_GaussianKernel_sim(mi_dis)     # miRNA-disease
    mi_gau_2 = calculate_GaussianKernel_sim(lnc_mi.T)     # miRNA-lncRNA
    mi_fun = getRNA_functional_sim(RNAlen=len(mi_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(mi_dis))
    mi_sim = RNA_fusion_sim(mi_gau_1,mi_gau_2,mi_fun)

    "disease similarity"
    dis_gau_1 = calculate_GaussianKernel_sim(lnc_dis.T)  # lncRNA-disease
    dis_gau_2 = calculate_GaussianKernel_sim(mi_dis.T)   # miRNA-disease
    dis_sim = dis_fusion_sim(dis_gau_1,dis_gau_2,dis_sem_sim)










