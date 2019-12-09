import numpy as np
from classify import read_node_label
from basic_functions import read_graph_as_matrix, evaluate_embeddings_with_split
from label_utils_functions import completely_imbalanced_split_train
from compared_methods.rsdne_light import RSNDE_light

def run_RSDNE(G, idx_train=None, label_list=None):
    # set alpha=0, our method would reduce to MFDW (Network representation learning with rich text information IJCAI15)
    U = RSNDE_light(G, idx_train, label_list, lowRank=200, alpha=0.1, lamb=0.1, lr=0.1, maxIter=100)
    U = np.array(U)
    vectors = {}
    for i, embedding in enumerate(U):
        vectors[str(i)] = embedding
    return vectors

def evaluate_RSNDE(vectors, X_train_idx, X_test_idx, Y_train, Y_test, Y_all):
    res = evaluate_embeddings_with_split(vectors, X_train_idx, Y_train, X_test_idx, Y_test, Y_all, testnum=5)
    print(res)

if __name__ == '__main__':
    import os
    datafile='citeseer'
    edge_file = os.path.join("datasets", datafile, "graph.txt")
    label_file = os.path.join("datasets", datafile, "group.txt") 
    single_label = True
    
    X, Y = read_node_label(label_file)
    G = read_graph_as_matrix(nodeids=X, edge_file=edge_file)
    
    removed_class= ['0', '1']
    X_train_idx, X_test_idx, Y_train, Y_test, X_train_cid_idx, Y_train_cid = completely_imbalanced_split_train(X, Y, train_precent=0.5, removed_class=removed_class)
    print('completely-imbalanced train number', len(X_train_cid_idx))

    vectors = run_RSDNE(G, X_train_cid_idx, Y_train_cid)
    res = evaluate_RSNDE(vectors, X_train_idx, X_test_idx, Y_train, Y_test, Y_all=Y)