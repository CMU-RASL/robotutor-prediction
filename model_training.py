from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from joblib import dump, load
import numpy as np
from multiprocessing import Pool
from os import listdir
from helper import find_model_split
from probability_propagating import run_models

def get_training_split(num_series, k=10, perc=0.7, cross_bool=True):
    np.random.seed(0)
    inds = np.arange(num_series)
    np.random.shuffle(inds)

    num_series = inds.shape[0]
    group_size = np.ceil(num_series/k).astype('int')
    group_inds = []
    for ii, ki in enumerate(range(k)):
        group_inds.append(inds[ii*group_size:min(ii*group_size+group_size,
                num_series)])

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

def fit_model(params):
    train_X, train_Y, train_T, start_val, end_val, filename, num_classes, \
            foldername, model_type, component_num, tot_model_num = params

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

        for class_num in range(num_classes):
            model = GaussianMixture(n_components=component_num)
            cur_X = flat_train_X[np.where(flat_train_Y == class_num)[0]]
            cur_Y = flat_train_Y[np.where(flat_train_Y == class_num)[0]]
            if cur_X.shape[0] <component_num + 4:
                cur_X = np.vstack((cur_X, np.random.rand(10, cur_X.shape[1]) ))
                cur_Y = np.hstack((cur_Y, class_num*np.ones((10))))
            model.fit(cur_X, cur_Y)

            dump(model, 'models//' + foldername + '//' + filename + str(class_num) + '.joblib')

            print('\t Fit Model {}/{}'.format(len(listdir('models//'+foldername)), tot_model_num))

def load_model(filename, foldername):
    model = load('models//'+foldername + '//' + filename)
    return model

def create_all_models(foldername, num_models, class_num_arr,
    num_workers, X, Ys, T, train_inds, model_bool,component_num):

    tot_model_num = [0,0]
    model_split = [np.empty((len(train_inds), num_models[0]+1)),
            np.empty((len(train_inds), num_models[1]+1))]
    for model_type in range(len(model_split)):
        for fold_ind in range(len(train_inds)):
            split = find_model_split([T[ii] for ii in train_inds[fold_ind] ],
                        num_models[model_type])
            model_split[model_type][fold_ind, :] = split
            tot_model_num[model_type] += len(split)-1
        tot_model_num[model_type] = tot_model_num[model_type]*class_num_arr[model_type]
    tot_model_num = tot_model_num[0] + tot_model_num[1]

    if model_bool:
        param_vec = []
        for model_type in range(len(model_split)):
            for fold_ind in range(len(train_inds)):
                for model_start_ind in range(model_split[model_type].shape[1]-1):
                    start_val = model_split[model_type][fold_ind, model_start_ind]
                    end_val = model_split[model_type][fold_ind, model_start_ind+1]
                    model_name = 'Modeltype_{}_Start_{}_End_{}_Fold_{}_Classnum_'.format(
                            model_type,start_val, end_val, fold_ind)

                    if not model_name in listdir('models//'+foldername):
                        params = ([X[ii] for ii in train_inds[fold_ind]],
                                  [Ys[model_type][ii] for ii in train_inds[fold_ind]],
                                  [T[ii] for ii in train_inds[fold_ind]],
                                  start_val, end_val, model_name,
                                  class_num_arr[model_type], foldername, model_type,
                                  component_num, tot_model_num)
                        param_vec.append(params)

        pool = Pool(processes=num_workers)
        pool.map(fit_model, param_vec)
        # for param in param_vec:
        #     fit_model(param)

    return model_split

def get_acc(model_foldername, modeltype_ind, model_split, fold_ind, k, Xk,
        Yk, Tk, A, B, class_weight, num_classes, plot_foldername, plot_bool,
        guess_bool=False, guess_acc_bool=True):

    cur_model_arr = [[],[]]
    for model_start_ind in range(model_split.shape[0]-1):
        #Load the models into the area
        start_val = model_split[model_start_ind]
        end_val = model_split[model_start_ind+1]
        cur_models = []
        for class_num in range(num_classes):
            model_name = 'Modeltype_{}_Start_{}_End_{}_Fold_{}_Classnum_{}.joblib'.format(
                    modeltype_ind, start_val, end_val, fold_ind, class_num)

            if not model_name in listdir('models//'+model_foldername):
                print('\t Could not find {}'.format(model_foldername+'//'+model_name))
                return
            else:
                model = load_model(model_name, model_foldername)
                cur_models.append(model)
        cur_model_arr[0].append((start_val, end_val))
        cur_model_arr[1].append(cur_models)

    #Run valid data through models
    if num_classes == 3:
        label = 'Feedback'
    else:
        label = 'Backbutton'
    # print('\t {} - Fold {}/{}'.format(label, fold_ind+1, k))
    acc, early = run_models(Xk, Yk, Tk, cur_model_arr,
                    A, B, class_weight, num_classes =
                    num_classes, guess_bool = guess_bool,
                    guess_acc_bool = guess_acc_bool,
                    plot_graphs = plot_bool,
                    plot_confusions = plot_bool, name = 'test',
                    img_name = plot_foldername + '//prob_test_', incr=None)

    return acc.flatten(), early.flatten()
