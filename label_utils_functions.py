import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection as sk_ms
from sklearn.preprocessing import LabelBinarizer
# support multi class

def get_class_set(labels):
    # labels [l, [c1, c2, ..]]
    # return：the labeled class set
    mydict = {}
    for y in labels:
        for label in y:
            mydict[label] = 1
    return mydict.keys()

def build_label_hash_list(idx_train, label_list, labeltype='intra'):
    '''
        Inputs:
            type: 'intra' means same labeled, 'inter' means diff labeled
        Return: 
            the hash map[label_c, node1, noid2, ...]
    '''
    class_set = get_class_set(label_list)
    mydict = {}
    for c in class_set:
        nodelist = [] 
        for node, labels in zip(idx_train, label_list):
            if( labeltype == 'intra' ):
                if( str(c) in labels ): nodelist.append(node)
            if( labeltype == 'inter' ):
                 if( str(c) not in labels): nodelist.append(node)
        mydict[c] = nodelist
    return mydict

def check_cover_all_class(Y_train, Y_all):
    # check if the train data cover all classes
    train_classes = get_class_set(Y_train)
    all_classes = get_class_set(Y_all)

    if(len(train_classes) == len(all_classes)): return True
    else: return False

def cover_all_class_split_train(X, Y, train_precent):
    """ We ensure every class has train/test instances
        Input: 
        Output: X_train [l, 1], X_test [n-l, 1]: the index node id list
                Y_train [ l, [c1, c2, ..]] , Y_test [n-l, [c1, c2, ..]]: the label of the cropossending node
    """
    training_size = int(train_precent * len(X))
    flag = False
    while flag is False:
        print('---- split once ------')
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train_idx = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test_idx = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        flag = check_cover_all_class(Y_train, Y)
    #print(type(X_train), type(Y_train))
    return X_train_idx, X_test_idx, Y_train, Y_test

def completely_imbalanced_split_train_single_label(X, Y, train_precent, removed_class):
    ''' 1) first split train/test part, 2) remove the labeled nodes in the removed_class
        Return: 6 list: X_train_idx [l], X_test_idx [n-l], Y_train [l, [c1,c2,...]], Y_test [n-l, [c1,c2,...]] are the common train\test split
                        X_train_cid_idx [l'], Y_train_cid [l', [c1,c2,...]] are the completely-imbalaced training data
    '''
    X_train_idx, X_test_idx, Y_train, Y_test = cover_all_class_split_train(X, Y, train_precent)
    idxlist = []
    for i in range(len(X_train_idx)):
        #print(set(Y_train[i]), set(removed_class), set(Y_train[i]) & set(removed_class))
        if( len(set(Y_train[i])&set(removed_class)) == 0 ):
            idxlist.append(i)
            
    X_train_cid_idx = [X_train_idx[i] for i in idxlist]
    Y_train_cid = [Y_train[i] for i in idxlist]
    return X_train_idx, X_test_idx, Y_train, Y_test, X_train_cid_idx, Y_train_cid

def completely_imbalanced_split_train(X, Y, train_precent, removed_class, single_label=True):
    if( single_label ): 
        return completely_imbalanced_split_train_single_label(X, Y, train_precent, removed_class)
