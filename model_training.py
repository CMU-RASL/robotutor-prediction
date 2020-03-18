from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
import numpy as np
from multiprocessing import Pool
import os

def get_training_split(num_series, k=4, perc=0.8, cross_bool=False):
    np.random.seed(0)
    inds = np.arange(num_series)
    np.random.shuffle(inds)

    if cross_bool:
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
    else:
        train_split = np.floor(perc*num_series).astype('int')
        train_ind = [np.arange(train_split)]
        test_ind = [np.arange(train_split,num_series)]

    return train_ind, test_ind

def fit_model(params):
    train_X, train_Y, train_T, start_val, end_val, filename, num_classes, foldername, model_type = params

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

    if len(X) > 0:
        flat_train_X = np.vstack(X)
        flat_train_Y = np.vstack(Y).flatten()

        class_weight= {}
        for ii in range(num_classes):
            class_weight[ii] = 0.0

        unique_elements, counts_elements = np.unique(flat_train_Y,
                                                     return_counts=True)
        tmp_x = np.zeros((1, flat_train_X.shape[1]))
        for ii in range(num_classes):
            if not ii in unique_elements:
                flat_train_X = np.vstack((flat_train_X, tmp_x))
                flat_train_Y = np.concatenate((flat_train_Y, np.array([ii])))

        unique_elements, counts_elements = np.unique(flat_train_Y,
                                                     return_counts=True)

        weights = counts_elements / flat_train_Y.shape[0]
        for label, weight in zip(unique_elements, weights):
            class_weight[label] = weight

        if model_type == 'SVC':
            model = SVC(probability=True, gamma='auto', class_weight=class_weight)
        if model_type == 'RandomForest':
            model = RandomForestClassifier(class_weight=class_weight)
        if model_type == 'DecisionTree':
            model = DecisionTreeClassifier(class_weight=class_weight)

        model.fit(flat_train_X, flat_train_Y)

        dump(model, foldername + '//' + filename)
        print('Fit', filename)

def load_model(filename, foldername):
    model = load(foldername + '//' + filename)
    return model

def create_all_models(foldername, model_split, k, cross_bool, class_num_arr,
    num_workers, X, Ys, T, train_inds, test_inds, model_type):

    pool = Pool(processes=num_workers)
    param_vec = []
    for model_ind in range(len(model_split)):
        for fold_ind in range(len(train_inds)):
            for model_start_ind in range(len(model_split[model_ind])-1):
                start_val = model_split[model_ind][model_start_ind]
                end_val = model_split[model_ind][model_start_ind+1]
                model_name = 'Model_{}_Start_{}_End_{}_Fold_{}.joblib'.format(model_ind,
                                    start_val, end_val, fold_ind)

                if not model_name in os.listdir(foldername):
                    params = (X[train_inds[fold_ind]],
                              Ys[model_ind][train_inds[fold_ind]],
                              T[train_inds[fold_ind]],
                              start_val, end_val, model_name,
                              class_num_arr[model_ind], foldername, model_type)
                    param_vec.append(params)

    print('Number of models {}'.format(len(param_vec)))
    pool.map(fit_model, param_vec)
