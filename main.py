import numpy as np
from os import mkdir, listdir
from model_training import get_training_split, create_all_models, get_acc
from helper import remove_readonly, filter_data, get_class_weight
from shutil import rmtree
from pickle import dump, load
from timeit import default_timer as timer

def main(k=10,data_name='dataset2',plot_bool=False,feature_set='all',
    col_to_remove='None'):

    start = timer()
    print('_________________')
    print('Setting Parameters:')

    #Hyperparameter arrays
    thresh_arr = np.arange(0.55, 1.0, 0.05)
    num_thresh = thresh_arr.shape[0]

    incr_arr = np.arange(0, 1, 0.02)
    num_incr = incr_arr.shape[0]

    num_model_arr = np.arange(1, 7).astype('int')
    num_model = num_model_arr.shape[0]

    max_depth_arr = np.arange(5, 50, 5).astype('int')
    # max_depth_arr = np.array([30])
    num_max_depth = max_depth_arr.shape[0]

    str_params = 'thresh_{:.4}_{:.4}_{:.4}_incr__{:.4}_{:.4}_{:.4}'.format(
        np.min(thresh_arr), np.max(thresh_arr), thresh_arr[1]-thresh_arr[0],
        np.min(incr_arr), np.max(incr_arr), incr_arr[1]-incr_arr[0])

    class_num_arr = [3, 2]
    modeltype_name_arr = ['Feedback', 'Backbutton']

    #Parameters
    num_workers = 3
    data_filename = data_name + '.pkl'

    #Load data
    with open(data_filename, 'rb') as f:
        data = load(f)

    #Filter data
    X, Y1, Y2, T, feat_names = data['X'], data['Y1'], data['Y2'], \
                                data['T'], data['feat_names']
    X, Y1, Y2, T, feat_names = filter_data(X, Y1, Y2, T, feat_names,
                                feature_set, col_to_remove)
    Ys = [Y1, Y2]

    #Training Split
    train_inds, test_inds = get_training_split(len(X), k)

    print('_________________')
    print('Cross-Validation:')

    test_accs = np.zeros((2, k, num_model, num_max_depth, num_thresh, num_incr))
    test_early = np.zeros((2, k, num_model, num_max_depth, num_thresh, num_incr))
    tot_num = 2*k*num_model*num_max_depth
    cur_num = 1

    for model_num_ind, model_num in enumerate(num_model_arr):
        for max_depth_ind, max_depth in enumerate(max_depth_arr):
            #Create Plotting Folder
            if col_to_remove == 'None':
                plot_foldername = 'plot_data_{}_folds_{}_modelnum_{}_maxdepth_{}_{}_features_{}'.format(
                                data_name, k, model_num, max_depth, str_params, feature_set)
            else:
                plot_foldername = 'plot_data_{}_folds_{}_modelnum_{}_maxdepth_{}_{}_features_{}_remove_{}'.format(
                                data_name, k, model_num, max_depth, str_params, feature_set, col_to_remove)
            if plot_bool:
                if plot_foldername in listdir('plot'):
                    rmtree(plot_foldername, onerror=remove_readonly)
                mkdir('plot//'+plot_foldername)

            #Model Folder
            if col_to_remove == 'None':
                model_foldername = 'model_data_{}_folds_{}_modelnum_{}_maxdepth_{}_features_{}'.format(
                                data_name, k, model_num, max_depth, feature_set)
            else:
                model_foldername = 'model_data_{}_folds_{}_modelnum_{}_maxdepth_{}_features_{}_remove_{}'.format(
                                data_name, k, model_num, max_depth, feature_set, col_to_remove)

            if model_foldername in listdir('models'):
                model_bool = False
            else:
                model_bool = True
                mkdir('models//'+model_foldername)

            #Create Models
            # print('\t Creating Models for Model Num {}/{}, Component Num {}/{}'.format(
            #             model_num_ind+1, len(num_model_arr), component_num_ind+1,
            #             len(num_component_arr)))
            model_split = create_all_models(model_foldername, [model_num,model_num],
                class_num_arr, num_workers, X, Ys, T, train_inds, model_bool, max_depth)

            #For each modeltype
            for modeltype_ind in range(len(model_split)):

                #Prob filename
                if col_to_remove == 'None':
                    filename = 'prob_data_{}_folds_{}_modeltype_{}_modelnum_{}_max_depth_{}_{}_features_{}.pkl'.format(
                                data_name, k, modeltype_ind, model_num, max_depth, str_params, feature_set)
                else:
                    filename = 'prob_data_{}_folds_{}_modeltype_{}_modelnum_{}_max_depth_{}_{}_features_{}_remove_{}.pkl'.format(
                                data_name, k, modeltype_ind, model_num, max_depth, str_params, feature_set, col_to_remove)

                #Load probability if exists
                if filename in listdir('prob'):
                    with open('prob//'+filename, 'rb') as f:
                        tmpdata = load(f)
                    test_accs[modeltype_ind, :, model_num_ind, max_depth_ind, :, :] = tmpdata['accs']
                    test_early[modeltype_ind, :, model_num_ind, max_depth_ind, :, :] = tmpdata['earliness']

                #Otherwise, cross-validation
                else:
                    for fold_ind in range(len(train_inds)):
                        Xk = [X[ii] for ii in test_inds[fold_ind]]
                        Yk = [Ys[modeltype_ind][ii] for ii in test_inds[fold_ind]]
                        Tk = [T[ii] for ii in test_inds[fold_ind]]
                        class_weight = get_class_weight([Ys[modeltype_ind][ii][-1] for ii in train_inds[fold_ind]],
                            class_num_arr[modeltype_ind])
                        test_accs[modeltype_ind, fold_ind, model_num_ind, max_depth_ind, :, :] ,\
                        test_early[modeltype_ind, fold_ind, model_num_ind, max_depth_ind, :, :] = get_acc(model_foldername,
                                        modeltype_ind, model_split[modeltype_ind][fold_ind,:],
                                        fold_ind, k, Xk, Yk, Tk, thresh_arr, incr_arr, class_weight,
                                        class_num_arr[modeltype_ind], plot_foldername, plot_bool)
                        print('{}/{}'.format(cur_num, tot_num))
                        cur_num+=1
                    #Save data
                    my_data = {'accs': test_accs[modeltype_ind, :, model_num_ind, max_depth_ind, :, :],
                            'earliness': test_early[modeltype_ind, :, model_num_ind, max_depth_ind, :, :]}
                    with open('prob//'+filename, 'wb') as output:
                        dump(my_data, output)

    print('_________________')
    print('Choosing Hyperparameters:')
    alpha = 1.0
    best_params = []
    best_inds = []
    for modeltype_ind in range(len(model_split)):

        #Average over classes and folds
        cur_test_accs = np.mean(test_accs[modeltype_ind, :, :, :, :], axis=0)
        cur_test_early = np.mean(test_early[modeltype_ind, :, :, :, :], axis=0)
        metric = alpha*cur_test_accs + (1-alpha)*cur_test_early
        print('{:.5f}'.format(np.max(cur_test_accs) - np.min(cur_test_accs)))
        #Choose best hyperparameters
        max_metric = np.max(metric)
        ind1, ind2, ind3, ind4 = np.where(metric == max_metric)
        best_model_num = num_model_arr[ind1]
        best_max_depth = max_depth_arr[ind2]
        best_thresh = thresh_arr[ind3]
        best_incr = incr_arr[ind4]
        print('')
        # print('\t {} - All Best Model Number'.format(modeltype_name_arr[modeltype_ind]), best_model_num)
        # print('\t {} - All Best Max Depth'.format(modeltype_name_arr[modeltype_ind]), best_max_depth)
        # print('\t {} - All Best Thresholds'.format(modeltype_name_arr[modeltype_ind]), np.round(best_thresh, decimals=2))
        # print('\t {} - All Best Increments'.format(modeltype_name_arr[modeltype_ind]), np.round(best_incr, decimals=2))

        tmp_ind = 0
        print('\t {} - Best Model Number {}, Best Max Depth {}, Best Threshold {:.5f}, Best Increment {:.5f}'.format(
                    modeltype_name_arr[modeltype_ind], best_model_num[tmp_ind], best_max_depth[tmp_ind], best_thresh[tmp_ind], best_incr[tmp_ind]))
        print('\t {} - For best parameters - Metric {:.5f}, Accuracy {:.5f}, Earliness {:.5f}'.format(
                    modeltype_name_arr[modeltype_ind], max_metric, cur_test_accs[ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind], ind4[tmp_ind]],
                    cur_test_early[ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind], ind4[tmp_ind]]))
        best_params.append((best_model_num[tmp_ind], best_max_depth[tmp_ind], best_thresh[tmp_ind], best_incr[tmp_ind]))
        best_inds.append((ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind], ind4[tmp_ind]))


    my_data = {'thresh_arr': thresh_arr, 'incr_arr': incr_arr, 'num_model_arr': num_model_arr,
        'max_depth_arr': max_depth_arr,
        'feat_names': feat_names, 'test_accs': test_accs,
        'test_early': test_early, 'best_inds': best_inds, 'best_params': best_params}

    if col_to_remove == 'None':
        result_filename = 'result//result_features_{}.pkl'.format(feature_set)
    else:
        result_filename = 'result//result_features_{}_remove_{}.pkl'.format(feature_set, col_to_remove)
    with open(result_filename, 'wb') as output:
        dump(my_data, output)

    end = timer()
    print('Total Minutes - {:.4}, Saved to - {}'.format((end - start)/60, result_filename))


if __name__ == '__main__':
    cols = ['Activity Ind', 'Video Time', 'Head Proximity', 'Head Orientation', 'Gaze Direction', 'Eye Aspect Ratio', 'Pupil Ratio', 'AU04', 'AU07', 'AU12', 'AU25', 'AU26', 'AU45', 'Progress', 'Picture Side', 'Activity Type', 'Activity Time']
    # main(feature_set='context')
    main(feature_set='all')
    # main(feature_set='face')
    # for col in cols:
    #     main(feature_set='all', col_to_remove=col)
