import numpy as np
from os import mkdir, listdir
from plotting import plot_accuracies, plot_rocs
from model_training import load_model, get_training_split, create_all_models
from probability_propagating import run_models
from helper import remove_readonly
from shutil import rmtree
from pickle import dump
from timeit import default_timer as timer

def main(ablation_col, max_depth):
    start = timer()
    #Parameters
    #model_split = np.array([[0, 30, 100, 300, 450, -1],
    #                        [0, 10, 20, 50, 200, -1]])
    model_split = np.array([[0, 30, 100, 300, 450, -1]])

    k = 4
    train_perc = 0.8
    num_workers = 4
    num_thresh = 6

    cross_bool = True
    plot_bool = False
    create_model_bool = True

    data_name = 'dataset1_vid'
    model_type = 'RandomForest'

    data_filename = data_name + '.npz'
    plot_foldername = 'plot_' + data_name + '_' + model_type + '_' + \
            str(int(cross_bool)) + '_' + ablation_col.replace(" ", "") + \
            '_' + str(max_depth)
    model_foldername = 'model_' + data_name + '_' + model_type + '_' + \
            str(int(cross_bool)) + '_' + ablation_col.replace(" ", "") + \
            '_' + str(max_depth)
    result_filename = 'result_' + data_name + '_' + model_type + '_' + \
            str(int(cross_bool)) + '_' + ablation_col.replace(" ", "") + \
            '_' + str(max_depth) + '.pkl'

    class_num_arr = [3, 2]
    headers = ['Head Proximity', 'Head Orientation', 'Gaze Direction',
                'Eye Aspect Ratio', 'Pupil Ratio', 'Progress', 'Picture Side',
                'Activity']

    if len(ablation_col) > 0:
        ablation_ind = headers.index(ablation_col)
    else:
        ablation_ind = -1

    #Load data
    data = np.load(data_filename, allow_pickle = True)
    X, Y1, Y2, T = data['X'], data['Y1'], data['Y2'], data['T']

    if ablation_ind >= 0:
        for xx in X:
            xx = np.delete(xx, ablation_ind, 1)

    Ys = [Y1, Y2]

    if not cross_bool:
        k = 1

    #Training Split
    train_inds, test_inds = get_training_split(len(X), k, train_perc,
            cross_bool)

    if create_model_bool:
        if model_foldername in listdir():
            rmtree(model_foldername, onerror=remove_readonly)
        mkdir(model_foldername)

        #Create all models - uncomment for creation of joblib files
        create_all_models(model_foldername, model_split, k, cross_bool,
            class_num_arr, num_workers, X, Ys, T, train_inds, test_inds,
            model_type, max_depth)

    #Create folders
    if plot_bool:
        if plot_foldername in listdir():
            rmtree(plot_foldername, onerror=remove_readonly)
        mkdir(plot_foldername)
        mkdir(plot_foldername + '/accuracy')
        mkdir(plot_foldername + '/roc')
        mkdir(plot_foldername + '/confusion')
        mkdir(plot_foldername + '/prob_train')
        mkdir(plot_foldername + '/prob_test')

    #Create threshold areas
    thresh_arr = np.linspace(0.5, 0.9, num_thresh).astype('float')

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
                model_name = 'Model_{}_Start_{}_End_{}_Fold_{}.joblib'.format(
                model_ind, start_val, end_val, fold_ind)

                if not model_name in listdir(model_foldername):
                    print('\t Could not find {}'.format(model_name))
                    return
                else:
                    model = load_model(model_name, model_foldername)
                cur_model_arr[0].append((start_val, end_val))
                cur_model_arr[1].append(model)
            print('\t Loaded Models for Fold {}/{}'.format(fold_ind+1, k))

            #For each threshold
            for thresh_ind, thresh in enumerate(thresh_arr):

                # #Run training data through models
                # img_name = 'Model_{}_Fold_{}_Thresh_{:.4f}_train'.format(
                #     model_ind, fold_ind, thresh)
                # fpr, tpr, acc = run_models(X[train_inds[fold_ind]],
                #                      Ys[model_ind][train_inds[fold_ind]],
                #                      T[train_inds[fold_ind]], cur_model_arr,
                #                      thresh, num_classes =
                #                      class_num_arr[model_ind],
                #                      plot_graphs = plot_bool,
                #                      plot_confusions=plot_bool, name = 'train',
                #                      img_name = plot_foldername +
                #                         '//prob_train//' + img_name)
                # train_fprs[model_ind][fold_ind, thresh_ind, :] = fpr
                # train_tprs[model_ind][fold_ind, thresh_ind, :] = tpr
                # train_accs[model_ind][fold_ind, thresh_ind, :] = acc

                #Run test data through models
                img_name = 'Model_{}_Fold_{}_Thresh_{:.4f}_test'.format(
                        model_ind, fold_ind, thresh)
                fpr, tpr, acc = run_models(X[test_inds[fold_ind]],
                                     Ys[model_ind][test_inds[fold_ind]],
                                     T[test_inds[fold_ind]], cur_model_arr,
                                     thresh, num_classes =
                                     class_num_arr[model_ind],
                                     plot_graphs = plot_bool,
                                     plot_confusions=plot_bool, name = 'test',
                                     img_name = plot_foldername +
                                        '//prob_test//' + img_name)
                test_fprs[model_ind][fold_ind, thresh_ind, :] = fpr
                test_tprs[model_ind][fold_ind, thresh_ind, :] = tpr
                test_accs[model_ind][fold_ind, thresh_ind, :] = acc

                print('\t\t Fold {}/{}, Thresh {}/{}'.format(fold_ind+1,
                      k,thresh_ind+1,num_thresh))

    #Create plots for accuracies and rocs
    # for model_ind in range(len(model_split)):
    #     img_name = plot_foldername + '//accuracy//' + 'Model_{}_train'.format(
    #             model_ind)
    #     if plot_bool:
    #         plot_accuracies(train_accs[model_ind], thresh_arr, 'train',
    #                 img_name)
    #
    #     img_name = plot_foldername + '//roc//' + 'Model_{}_train'.format(
    #             model_ind)
    #     if plot_bool:
    #         plot_rocs(train_fprs[model_ind], train_tprs[model_ind], thresh_arr,
    #             'train',img_name)
    #
    #     img_name = plot_foldername + '//accuracy//' + 'Model_{}_test'.format(
    #             model_ind)
    #     if plot_bool:
    #         plot_accuracies(test_accs[model_ind], thresh_arr, 'test',img_name)
    #
    #     img_name = plot_foldername + '//roc//' + 'Model_{}_test'.format(
    #             model_ind)
    #     if plot_bool:
    #         plot_rocs(test_fprs[model_ind], test_tprs[model_ind], thresh_arr,
    #                 'test',img_name)

    my_data = {'train_fprs':train_fprs, 'train_tprs':train_tprs,
        'train_accs':train_accs, 'test_fprs':test_fprs, 'test_tprs':test_tprs,
        'test_accs':test_accs}
    output = open(result_filename, 'wb')
    dump(my_data, output)
    output.close()

    end = timer()
    print('Total Minutes: {:.4}'.format((end - start)/60))

if __name__ == '__main__':
    headers = ['', 'Head Proximity', 'Head Orientation', 'Gaze Direction',
                'Eye Aspect Ratio', 'Pupil Ratio', 'Progress', 'Picture Side',
                'Activity']
    depth_arr = np.linspace(1, 50, 10).astype('int')
    for ii, header in enumerate(headers):
        main(header, 100)
        print('Finished {}/{}'.format(ii, len(headers)))
