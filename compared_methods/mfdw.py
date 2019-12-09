import numpy as np
from compared_methods.matrix_decomposition import mf
from sklearn.preprocessing import normalize

# MF_DEEPWALK DeepWalk as Matrix Factorization
# Network representation learning with rich text information IJCAI15

def graph_of_deepwalk(G):
    '''Calculate: G = (G+G^2)/2
       Inputs: G - np[n,n], note the in the MFDW sparse matrix is not supported
    '''
    G = normalize(G, norm='l1', axis=1)
    G = (G + np.dot(G, G))/2
    return G


def MFDW(G, d=200, lamb=0.1, lr=0.01, maxIter=100):
    # MF_DEEPWALK DeepWalk as Matrix Factorization Zhiyuan Liu IJCAI15
    G = graph_of_deepwalk(G)
    #U, V = matrix_decomposition.mf(G, d=d, lr=lr, maxIter=maxIter)
    U, V = mf(G, m=d, lamb=lamb, lr=lr, maxIter=maxIter)
    return U