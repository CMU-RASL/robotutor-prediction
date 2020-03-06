import numpy as np
import os
from plotting import plot_accuracies, plot_rocs
from model_training import load_model, get_training_split, create_all_models
from probability_propagating import run_models

def main():
    #Parameters
    model_split = np.array([[0, 30, 100, 300, 450, -1],
                            [0, 10, 20, 50, 200, -1]])
    k = 4
    class_num_arr = [3, 2]
    data_filename = 'dataset1_log.npz'
    num_workers = 5
    num_thresh = 5

    #Load data
    data = np.load(data_filename, allow_pickle = True)
    X, Y1, Y2, T = data['X'], data['Y1'], data['Y2'], data['T']
    Ys = [Y1, Y2]

    #Training Split
    train_inds, test_inds = get_training_split(X, Y1, Y2, T, k)

    #Create all models - uncomment for creation of joblib files
    #create_all_models(model_split, k, class_num_arr, num_workers, X, Ys, T, train_inds, test_inds)

    #Create threshold areas
    thresh_arr = np.linspace(0, 1, num_thresh).astype('float')

    #Save the train-test split into txt file
    filename = 'train_test_split.txt'
    f = open(filename, 'w+')
    for ii, test_ind in enumerate(test_inds):
        name = '-'.join(map(str, test_ind))
        f.write('{:02} \t {}\n'.format(ii, name))
    f.close()

    #Initialize fpr, tpr, and acc arrays
    train_fprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    train_tprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    train_accs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_fprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_tprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_accs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]

    #For each of the two model types
    for model_ind in range(len(model_split)):
        print('Model_{}/{}'.format(model_ind+1, len(model_split)))
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

                if not model_name in os.listdir('models'):
                    print('\t Could not find {}'.format(model_name))
                    #return
                else:
                    model = load_model(model_name)
                cur_model_arr[0].append((start_val, end_val))
                cur_model_arr[1].append(model)
            print('\t Loaded Models for Fold {}/{}'.format(fold_ind+1, k))

            #For each threshold
            for thresh_ind, thresh in enumerate(thresh_arr):

                #Run training data through models

                img_name = 'Model_{}_Fold_{}_Thresh_{}_train'.format(model_ind, fold_ind, thresh)
                fpr, tpr, acc = run_models(X[train_inds[fold_ind]],
                                     Ys[model_ind][train_inds[fold_ind]],
                                     T[train_inds[fold_ind]], cur_model_arr,
                                     thresh, num_classes =
                                     class_num_arr[model_ind],
                                     plot_graphs = False,
                                     plot_confusions=False, name = 'train',
                                     img_name = img_name)
                train_fprs[model_ind][fold_ind, thresh_ind, :] = fpr
                train_tprs[model_ind][fold_ind, thresh_ind, :] = tpr
                train_accs[model_ind][fold_ind, thresh_ind, :] = acc

                #Run test data through models
                img_name = 'Model_{}_Fold_{}_Thresh_{}_test'.format(model_ind, fold_ind, thresh)
                fpr, tpr, acc = run_models(X[test_inds[fold_ind]],
                                     Ys[model_ind][test_inds[fold_ind]],
                                     T[test_inds[fold_ind]], cur_model_arr,
                                     thresh, num_classes =
                                     class_num_arr[model_ind],
                                     plot_graphs = False,
                                     plot_confusions=False, name = 'test',
                                     img_name = img_name)
                test_fprs[model_ind][fold_ind, thresh_ind, :] = fpr
                test_tprs[model_ind][fold_ind, thresh_ind, :] = tpr
                test_accs[model_ind][fold_ind, thresh_ind, :] = acc
                print('\t\t Fold {}/{}, Thresh {}/{}'.format(fold_ind+1,
                      k,thresh_ind+1,num_thresh))

    #Create plots for accuracies and rocs
    for model_ind in range(len(model_split)):
        img_name = 'Model_{}_train'.format(model_ind)
        plot_accuracies(train_accs[model_ind], thresh_arr, 'train',img_name,plot_bool=False)
        plot_rocs(train_fprs[model_ind], train_tprs[model_ind], thresh_arr,'train',img_name,plot_bool=False)

        img_name = 'Model_{}_test'.format(model_ind)
        plot_accuracies(test_accs[model_ind], thresh_arr, 'test',img_name,plot_bool=False)
        plot_rocs(test_fprs[model_ind], test_tprs[model_ind], thresh_arr, 'test',img_name, plot_bool=False)

if __name__ == '__main__':
    main()
