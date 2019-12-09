import numpy as np
from compared_methods.label_relax_single import build_Ls_matrix, build_Lw_matrix
from compared_methods.mfdw import graph_of_deepwalk

def getCost_RSDNE(G, U, V, Ls, Lw, alpha, lamb):
    # Get the cost: J = |G-UV|^2 + alpha*( tr(U'LsU) + tr(U'LwU) ) + lambda(|U|^2 + |V|^2)
    # Inputs: G[n,n], U[n,d], v[d,n], L[n,n]
    UV = np.dot(U, V) 
    cost1 = np.sum((G - UV)**2)
    cost21 = alpha*np.trace( np.linalg.multi_dot([U.T, Ls, U]) )
    cost22 = alpha*np.trace( np.linalg.multi_dot([U.T, Lw, U]) )
    cost3 = lamb*np.trace(np.matmul(U.T, U)) 
    cost4 = lamb*np.trace(np.matmul(V.T, V)) 

    #print('cost1 %f, cost21 %f, cost22 %f, cost3 %f, cost4 %f:'%(cost1, cost21, cost22, cost3, cost4)) 
    res = cost1 + cost21 + cost22 + cost3 + cost4
    return res

def RSNDE_light(G, idx_train, label_list, lowRank, alpha=0, lamb=0, lr=0.1, maxIter = 50):
    ''' RSDNE: Relaxed Simi-larity and Dissimilarity Network Embedding AAAI18
        min(U,V) J = |G-UV|^2 + alpha*|U' (Ls+Lw) U|^2 + lambda*(|U|^2 + |V|^2)
        Inputs: G - [n,n] network strucutre, the deep walk matrix form of G
                idx_train: train node ids
                label_list: [l, 1] only support single label
    '''
    G = graph_of_deepwalk(G)
    #G, U, V = MFDW_G_U_V(G, d=200, lamb=0, lr=0.05, maxIter=100)
    #return U
    # Random initialization:
    node_num, nouse = G.shape
    U = np.random.random([node_num, lowRank]) / node_num
    V = np.random.random([lowRank, node_num]) / node_num

    Ls = build_Ls_matrix(node_num, idx_train, label_list, orgk=5)
    Lw = build_Lw_matrix(G, idx_train, label_list)
    Ls = Ls.todense()
    
    L = Ls + Lw
    L = np.asarray(L)
    #Gradient Descent:
    trainRMSE_old = getCost_RSDNE(G, U, V, Ls, Lw, alpha, lamb)
    print('cost at iteration start :', trainRMSE_old)
    for i in range(maxIter):
        temp = 2*alpha*np.dot(L, U)
        #print(type(temp), temp.dtype)
        dU = 2*(-np.dot(G, V.T) + np.linalg.multi_dot([U, V, V.T]) + lamb*U) + temp
        U = U - lr*dU
        trainRMSE_new = getCost_RSDNE(G, U, V, Ls, Lw, alpha, lamb)
        #print('after update U :', trainRMSE_new)

        dV = 2*(-np.dot(U.T, G) + np.linalg.multi_dot([U.T, U, V]) + lamb*V)
        V = V - lr*dV 
        trainRMSE_new = getCost_RSDNE(G, U, V, Ls, Lw, alpha, lamb)
        #print('after update V :', trainRMSE_new)
        
        if(i%1 == 0): 
            print('cost at iteration %d : %f' % (i, trainRMSE_new))

    return U