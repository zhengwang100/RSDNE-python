from statistics import mean 
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from classify import Classifier, read_node_label
from label_utils_functions import completely_imbalanced_split_train

def evaluate_embeddings_with_split(vectors, X_train, Y_train, X_test, Y_test, Y_all, testnum=10):
    print("Training classifier with the pre-defined split setting...")
    #clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf = Classifier(vectors=vectors, clf=SVC(probability=True))
    micro_list = []; macro_list = []
    for i in range(testnum):
        res = clf.evaluate_with_fixed_split(X_train, Y_train, X_test, Y_test, Y_all)

        micro_list.append(res['micro'])
        macro_list.append(res['macro'])
    return mean(micro_list), mean(macro_list)

def symmetrize(a):
    #return a + a.T - sp.diags(a.diagonal())
    return a + a.T - np.diag(a.diagonal())
    
def read_graph_as_matrix(nodeids, edge_file):
    ''' Read a symmetric adjacency matrix from a file 
        Input: nodeids: [1,2,3,4,...]
        Return: the sparse adjacency matrix
    '''
    idx = np.array(nodeids, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(edge_file, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    print('origial input G', sp.coo_matrix.count_nonzero(adj))
    adj = symmetrize(adj)
    return adj
