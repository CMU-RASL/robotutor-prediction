from sklearn.svm import SVC
from joblib import dump, load
import numpy as np

def get_training_split(X, Y1, Y2, T, k=4, random = False):
    
    num_series = len(X)

    inds = [3, 36, 21, 14, 35,  6,  4,  8, 16, 15, 24,  2,  9, 31, 
            37, 18, 26, 13,  5, 11, 17, 28,  0,  1, 34, 32, 30, 25, 
            12, 20, 33, 19, 10,  7, 29, 27, 23, 22]
    
    group_size = np.ceil(num_series/k).astype('int')
    group_inds = []
    for ii, ki in enumerate(range(k)):
        group_inds.append(inds[ii*group_size:min(ii*group_size+group_size,num_series)])
    
    train_ind = []
    test_ind = []
    for ii in range(k):
        tmp_inds = []
        for jj in range(k):
            if ii == jj:
                test_ind.append(group_inds[ii])
            else:
                tmp_inds.extend(group_inds[jj])
        train_ind.append(tmp_inds)
    
    return train_ind, test_ind

def fit_model(train_X, train_Y, train_T, start_val, end_val, filename, \
              num_classes):
    
    X = []
    Y = []
    
    for ii in range(len(train_X)):
        start_ind = np.where(train_T[ii] >= start_val)[0]
        if end_val == -1:
            end_ind = np.where(train_T[ii] < train_T[ii][-1])[0]
        else:
            end_ind = np.where(train_T[ii] < end_val)[0]
        if start_ind.shape[0] > 0 and end_ind.shape[0] > 0:    
            X.append(train_X[ii][start_ind[0]:end_ind[-1],:])
            Y.append(train_Y[ii][start_ind[0]:end_ind[-1]])
    
    flat_train_X = np.vstack(X)
    flat_train_Y = np.vstack(Y).flatten()
    
    class_weight= {}
    for ii in range(num_classes):
        class_weight[ii] = 0.0
    
    unique_elements, counts_elements = np.unique(flat_train_Y, 
                                                 return_counts=True)
    
    weights = counts_elements / flat_train_Y.shape[0]
    for label, weight in zip(unique_elements, weights):
        class_weight[label] = weight
    model = SVC(probability=True, gamma='auto', class_weight=class_weight)
    model.fit(flat_train_X, flat_train_Y)
    
    dump(model, 'models/' + filename)
    
    return model


def load_model(filename):
    model = load('models/' + filename)
    return model