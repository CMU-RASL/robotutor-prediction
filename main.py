import numpy as np
import os
from plotting import plot_accuracies, plot_rocs
from model_training import load_model, get_training_split, create_all_models
from probability_propagating import run_models
from helper import remove_readonly
import shutil
import pickle

def main():
    #Parameters
    #model_split = np.array([[0, 30, 100, 300, 450, -1],
    #                        [0, 10, 20, 50, 200, -1]])
    model_split = np.array([[0, 30, 100, 300, 450, -1]])

    k = 4
    train_perc = 0.8
    num_workers = 4
    num_thresh = 11

    cross_bool = True
    plot_bool = False
    create_model_bool = True

    data_name = 'dataset1_vid'
    model_type = 'SVC'

    data_filename = data_name + '.npz'
    plot_foldername = 'plot_' + data_name + '_' + model_type + '_' + str(int(cross_bool))
    model_foldername = 'model_' + data_name + '_' + model_type + '_' + str(int(cross_bool))
    result_filename = 'result_' + data_name + '_' + model_type + '_' + str(int(cross_bool)) + '.pkl'
    class_num_arr = [3, 2]

    #Load data
    data = np.load(data_filename, allow_pickle = True)
    X, Y1, Y2, T = data['X'], data['Y1'], data['Y2'], data['T']

    Ys = [Y1, Y2]

    if not cross_bool:
        k = 1

    #Training Split
    train_inds, test_inds = get_training_split(len(X), k, train_perc, cross_bool)

    if create_model_bool:
        if model_foldername in os.listdir():
            shutil.rmtree(model_foldername, onerror=remove_readonly)
        os.mkdir(model_foldername)

        #Create all models - uncomment for creation of joblib files
        create_all_models(model_foldername, model_split, k, cross_bool,
            class_num_arr, num_workers, X, Ys, T, train_inds, test_inds, model_type)

    #Create folders
    if plot_foldername in os.listdir():
        shutil.rmtree(plot_foldername, onerror=remove_readonly)
    os.mkdir(plot_foldername)
    os.mkdir(plot_foldername + '/accuracy')
    os.mkdir(plot_foldername + '/roc')
    os.mkdir(plot_foldername + '/confusion')
    os.mkdir(plot_foldername + '/prob_train')
    os.mkdir(plot_foldername + '/prob_test')

    #Create threshold areas
    thresh_arr = np.linspace(0, 1, num_thresh).astype('float')

    #Initialize fpr, tpr, and acc arrays
    train_fprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    train_tprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    train_accs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_fprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_tprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_accs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]

    #For each of the two model types
    for model_ind in range(len(model_split)):
        print('Model {}/{}'.format(model_ind+1, len(model_split)))
        #For each fold
        for fold_ind in range(len(train_inds)):
            #For each model in model split
            cur_model_arr = [[],[]]
            for model_start_ind in range(len(model_split[model_ind])-1):

                #Load the models into the area
                start_val = model_split[model_ind][model_start_ind]
                end_val = model_split[model_ind][model_start_ind+1]
                model_name = 'Model_{}_Start_{}_End_{}_Fold_{}.joblib'.format(model_ind,
                                    start_val, end_val, fold_ind)

                if not model_name in os.listdir(model_foldername):
                    print('\t Could not find {}'.format(model_name))
                    return
                else:
                    model = load_model(model_name, model_foldername)
                cur_model_arr[0].append((start_val, end_val))
                cur_model_arr[1].append(model)
            print('\t Loaded Models for Fold {}/{}'.format(fold_ind+1, k))

            #For each threshold
            for thresh_ind, thresh in enumerate(thresh_arr):

                #Run training data through models
                img_name = 'Model_{}_Fold_{}_Thresh_{:.4f}_train'.format(model_ind, fold_ind, thresh)
                fpr, tpr, acc = run_models(X[train_inds[fold_ind]],
                                     Ys[model_ind][train_inds[fold_ind]],
                                     T[train_inds[fold_ind]], cur_model_arr,
                                     thresh, num_classes =
                                     class_num_arr[model_ind],
                                     plot_graphs = plot_bool,
                                     plot_confusions=plot_bool, name = 'train',
                                     img_name = plot_foldername + '//prob_train//' + img_name)
                train_fprs[model_ind][fold_ind, thresh_ind, :] = fpr
                train_tprs[model_ind][fold_ind, thresh_ind, :] = tpr
                train_accs[model_ind][fold_ind, thresh_ind, :] = acc

                #Run test data through models
                img_name = 'Model_{}_Fold_{}_Thresh_{:.4f}_test'.format(model_ind, fold_ind, thresh)
                fpr, tpr, acc = run_models(X[test_inds[fold_ind]],
                                     Ys[model_ind][test_inds[fold_ind]],
                                     T[test_inds[fold_ind]], cur_model_arr,
                                     thresh, num_classes =
                                     class_num_arr[model_ind],
                                     plot_graphs = plot_bool,
                                     plot_confusions=plot_bool, name = 'test',
                                     img_name = plot_foldername + '//prob_test//' + img_name)
                test_fprs[model_ind][fold_ind, thresh_ind, :] = fpr
                test_tprs[model_ind][fold_ind, thresh_ind, :] = tpr
                test_accs[model_ind][fold_ind, thresh_ind, :] = acc
                print('\t\t Fold {}/{}, Thresh {}/{}'.format(fold_ind+1,
                      k,thresh_ind+1,num_thresh))

    #Create plots for accuracies and rocs
    for model_ind in range(len(model_split)):
        img_name = plot_foldername + '//accuracy//' + 'Model_{}_train'.format(model_ind)
        plot_accuracies(train_accs[model_ind], thresh_arr, 'train',img_name,plot_bool=plot_bool)

        img_name = plot_foldername + '//roc//' + 'Model_{}_train'.format(model_ind)
        plot_rocs(train_fprs[model_ind], train_tprs[model_ind], thresh_arr,'train',img_name,plot_bool=plot_bool)

        img_name = plot_foldername + '//accuracy//' + 'Model_{}_test'.format(model_ind)
        plot_accuracies(test_accs[model_ind], thresh_arr, 'test',img_name,plot_bool=plot_bool)

        img_name = plot_foldername + '//roc//' + 'Model_{}_test'.format(model_ind)
        plot_rocs(test_fprs[model_ind], test_tprs[model_ind], thresh_arr, 'test',img_name, plot_bool=plot_bool)

    my_data = {'train_fprs':train_fprs, 'train_tprs':train_tprs,
        'train_accs':train_accs, 'test_fprs':test_fprs, 'test_tprs':test_tprs,
        'test_accs':test_accs}
    output = open(result_filename, 'wb')
    pickle.dump(my_data, output)
    output.close()


if __name__ == '__main__':
    main()
