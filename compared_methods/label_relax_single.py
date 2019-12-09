import sys
import random
import numpy as np
from scipy.sparse import csr_matrix, csgraph, coo_matrix
from label_utils_functions import build_label_hash_list

def build_node_pairs_by_single_label(idx_train, label_list, orgk=5, labeltype='intra'):
    ''' Get the intra or inter labeled nodes (Support multi-label, copy from label_relax_single.py) 
        Input: 
            idx_train: train node ids
            label_list: [l, [c1, c2, ..]] support multi label
            orgk: the intra-class neighbor number
            labeltype: 'intra' means must link and 'inter' means cannot link
        Return: node_list [l*orgk, 2] like [[node_i, node_j], ...]
    '''
    mydict = build_label_hash_list(idx_train, label_list, labeltype=labeltype)
    
    print('------ building node_pairs (neighbor) list ------', labeltype)
    node_pairs = []
    for node, labels in zip(idx_train, label_list):
        nodelist = []
        for label in labels:
            nodelist.extend(mydict[label])
        k = min(orgk, int(len(nodelist)/2))
        neighlist = random.sample(nodelist, k)
        for neighbor in neighlist:
            if(node == neighbor): continue 
            node_pairs.append([int(node), int(neighbor)])
    return node_pairs

def build_Ls_matrix(node_num, idx_train, label_list, orgk=5):
    node_pairs = build_node_pairs_by_single_label(idx_train, label_list, orgk=5, labeltype='intra')
    edges = np.array(node_pairs, dtype=np.int32)
    S = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(node_num, node_num),
                        dtype=np.float32)
    #print(type(S), S.count_nonzero())
    
    temp = (S + S.T)/2.0
    Ls = csgraph.laplacian(temp, normed=False) # L = D - temp    
    #print(type(Ls), Ls.count_nonzero())
    return Ls

def build_Lw_matrix(W, idx_train, label_list):
    '''Input:
            W: the sparse matrix
            idx_train: train node ids
            label_list: [l, [c1, c2, ..]] support multi label
        Return: Lw = D - (W+W')/2
    '''
    print('input G', type(W), np.count_nonzero(W))
    mydict = {}
    for nodeid, labels in zip(idx_train, label_list):
        mydict[int(nodeid)] = labels
    # set the value = 0, if two nodes are different labeled
    rows, cols = W.nonzero()
    for row, col in zip(rows, cols):
        if( (row in mydict) and (col in mydict) and (row!=col) ):
            if( len(set(mydict[row])&set(mydict[col])) == 0 ):
                W[row, col] = 0
                W[col, row] = 0
    print('dissimilar type G', type(W), np.count_nonzero(W))
    temp = (W + W.T)/2.0
    Lw = csgraph.laplacian(temp, normed=False) # L = D - temp
    
    return Lw
        
if __name__ == '__main__':
    import os
    from classify import read_node_label
    from label_utils_functions import completely_imbalanced_split_train
    from basic_functions import get_data_file, read_graph_as_matrix
    edge_file, label_file, feature_file = get_data_file()
    X, Y = read_node_label(label_file)

    removed_class = ['0', '1']
    X_train_idx, X_test_idx, Y_train, Y_test, X_train_cid_idx, Y_train_cid = completely_imbalanced_split_train(X, Y, 0.1, removed_class)
    print('train number', len(X_train_cid_idx))

    '''node_num = len(X_train_idx) + len(X_test_idx)
    print('node num:', node_num)
    Ls = build_Ls_matrix(node_num, X_train_cid_idx, Y_train_cid, orgk=5)
    print(Ls)'''

    nodeids, nouse = read_node_label(label_file)
    G = read_graph_as_matrix(nodeids, edge_file)
    Lw = build_Lw_matrix(G, X_train_cid_idx, Y_train_cid)
    print(Lw)

    '''
    embeddings = np.array([[1,2], [1,1], [3,3], [4,4], [-1,-1], [-2,-2], [-3,-3], [-4,-4]])
    print(embeddings)

    train_idxs = [0,1,2,4,5,6]
    train_labels = [0,0,0,1,1,1]

    hlist_1, hlist_2 = build_embedding_pairs_by_label(node_trainids, Y_train, embeddings, k=5, labeltype='intra')

    print(intra_list)
    '''