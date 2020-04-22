import numpy as np
from os import mkdir, listdir
from plotting import plot_accuracies, plot_rocs
from model_training import load_model, get_training_split, create_all_models
from probability_propagating import run_models
from helper import remove_readonly, filter_data, get_class_weight
from shutil import rmtree
from pickle import dump, load
from timeit import default_timer as timer

def main(num_models=4,k=4,data_name='dataset1_vid',incr=0.05):
    start = timer()

    #Parameters
    num_workers = 3
    num_thresh = 8

    #Create threshold areas
    thresh_arr = np.linspace(0.6, 1.0, num_thresh).astype('float')

    plot_bool = False
    model_bool = True

    data_filename = data_name + '.pkl'
    plot_foldername = 'plot_{}_folds_{}_model_num_{}_incr_{}'.format(data_name, k, num_models, incr)
    model_foldername = 'model_{}_folds_{}_model_num_{}_incr_{}'.format(data_name, k, num_models, incr)
    result_filename = 'result_{}_folds_{}_model_num_{}_incr_{}.pkl'.format(data_name, k, num_models, incr)

    class_num_arr = [3, 2]

    #Load data
    with open(data_filename, 'rb') as f:
        data = load(f)

    X, Y1, Y2, T, feat_names = data['X'], data['Y1'], data['Y2'], data['T'], data['feat_names']
    X, Y1, Y2, T, feat_names = filter_data(X, Y1, Y2, T, feat_names)

    Ys = [Y1, Y2]

    #Training Split
    train_inds, test_inds = get_training_split(len(X), k)

    if model_bool:
        if model_foldername in listdir():
            rmtree(model_foldername, onerror=remove_readonly)
        mkdir(model_foldername)

    #Create all models - uncomment for creation of joblib files
    model_split = create_all_models(model_foldername, num_models, k,
        class_num_arr, num_workers, X, Ys, T, train_inds, test_inds, model_bool)

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

    #Initialize acc arrays
    train_accs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_accs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_thresh_not_reached = [np.empty((k, num_thresh, 2)), np.empty((k, num_thresh, 2))]

    #For each of the two model types
    for model_ind in range(len(model_split)):
        print('Model {}/{}'.format(model_ind+1, len(model_split)))
        #For each fold
        for fold_ind in range(len(train_inds)):
            #For each model in model split
            cur_model_arr = [[],[]]
            for model_start_ind in range(num_models):

                #Load the models into the area
                start_val = model_split[model_ind][fold_ind,model_start_ind]
                end_val = model_split[model_ind][fold_ind,model_start_ind+1]
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
            # train_accs[model_ind][fold_ind, thresh_ind, :] = acc
            # print('train')

            #Run test data through models
            class_weight = get_class_weight(
                [Ys[model_ind][ii][-1] for ii in train_inds[fold_ind]],
                class_num_arr[model_ind])
            img_name = 'Model_{}_Fold_{}_test'.format(
                    model_ind, fold_ind)
            acc, num, den = run_models(
                            [X[ii] for ii in test_inds[fold_ind]],
                            [Ys[model_ind][ii] for ii in test_inds[fold_ind]],
                            [T[ii] for ii in test_inds[fold_ind]], cur_model_arr,
                            thresh_arr, class_weight, num_classes =
                            class_num_arr[model_ind],
                            plot_graphs = plot_bool,
                            plot_confusions=plot_bool, name = 'test',
                            img_name = plot_foldername +
                            '//prob_test//' + img_name, incr=incr)
            test_accs[model_ind][fold_ind, :, :] = acc
            test_thresh_not_reached[model_ind][fold_ind, :, 0] = num
            test_thresh_not_reached[model_ind][fold_ind, :, 1] = den
            print('\t Fold {}/{} Complete'.format(fold_ind+1,k))

    #Create plots for accuracies and rocs
    # for model_ind in range(len(model_split)):
    #     img_name = plot_foldername + '//accuracy//' + 'Model_{}_train'.format(
    #             model_ind)
    #     if plot_bool:
    #         plot_accuracies(train_accs[model_ind], thresh_arr, 'train',
    #                 img_name)
    #     img_name = plot_foldername + '//accuracy//' + 'Model_{}_test'.format(
    #             model_ind)
    #     if plot_bool:
    #         plot_accuracies(test_accs[model_ind], thresh_arr, 'test',img_name)

    my_data = {'train_accs':train_accs,'test_accs':test_accs,
        'thresh_arr': thresh_arr, 'feat_names': feat_names,
        'test_thresh_not_reached': test_thresh_not_reached}
    with open(result_filename, 'wb') as output:
        dump(my_data, output)

    if model_foldername in listdir():
        rmtree(model_foldername, onerror=remove_readonly)

    end = timer()
    print('Total Minutes: {:.4}'.format((end - start)/60))


if __name__ == '__main__':
    # headers = ['Head Proximity', 'Head Orientation', 'Gaze Direction',
    #         'Eye Aspect Ratio', 'Pupil Ratio', 'AU04', 'AU07', 'AU12', 'AU25',
    #         'AU26', 'AU45', 'Progress', 'Picture Side', 'Activity']
    #for ii, header in enumerate(headers):+
    # for num_models in np.arange(4,7):
    #     for incr in np.linspace(0.005, 0.5, 15):
    #         main(num_models=num_models,k=4,data_name = 'dataset1_vid',incr=incr)
    main(num_models=4,k=4,data_name='dataset1_vid',incr=0.05)
