import numpy as np
from os import mkdir, listdir
from model_training import get_training_split, create_all_models, get_acc
from helper import remove_readonly, filter_data, get_class_weight
from shutil import rmtree
from pickle import dump, load
from timeit import default_timer as timer

def main(data_name='dataset2_guess', k = 4, col_to_remove='None', feature_set='all', num_workers=3):

    start = timer()
    print('_________________')
    print('Create Models:')

    guess_bool = True
    guess_acc_bool = True

    #Hyperparameter arrays
    thresh_arr = np.arange(0.65, 0.85, 0.05)
    num_thresh = thresh_arr.shape[0]

    num_model_arr = np.arange(1, 3).astype('int')
    num_model = num_model_arr.shape[0]

    num_component_arr = np.arange(6, 7).astype('int')
    num_components = num_component_arr.shape[0]

    str_params = 'thresh_{:.4}_{:.4}_{:.4}_guess2'.format(np.min(thresh_arr),
                np.max(thresh_arr), thresh_arr[1]-thresh_arr[0])

    class_num_arr = [3, 2]
    modeltype_name_arr = ['Feedback', 'Backbutton']

    #Load data
    with open(data_name + '.pkl', 'rb') as f:
        data = load(f)

    #Filter data
    X, Y1, Y2, T, feat_names = data['X'], data['Y1'], data['Y2'], \
                                data['T'], data['feat_names']
    X, Y1, Y2, T, feat_names = filter_data(X, Y1, Y2, T, feat_names,
                                feature_set, col_to_remove)
    Ys = [Y1, Y2]

    #Training Split
    train_inds, test_inds = get_training_split(len(X), k)

    test_accs = np.zeros((2, k, num_model, num_components, num_thresh))
    test_early = np.zeros((2, k, num_model, num_components, num_thresh))
    tot_num = 2*k*num_model*num_components
    cur_num = 1

    for model_num_ind, model_num in enumerate(num_model_arr):
        for component_num_ind, component_num in enumerate(num_component_arr):
            model_foldername = 'model_data_{}_folds_{}_modelnum_{}_componentnum_{}_features_{}'.format(
                                data_name, k, model_num, component_num, feature_set)

            if model_foldername in listdir('models'):
                model_bool = False
            else:
                model_bool = True
                mkdir('models//'+model_foldername)

            model_split = create_all_models(model_foldername, [model_num,model_num],
                class_num_arr, num_workers, X, Ys, T, train_inds, model_bool, component_num)

            #For each modeltype
            for modeltype_ind in range(len(model_split)):

                filename = 'prob_data_{}_folds_{}_modeltype_{}_modelnum_{}_componentnum_{}_{}_features_{}_guessbool_{}_guessaccbool_{}.pkl'.format(
                            data_name, k, modeltype_ind, model_num, component_num, str_params, feature_set, guess_bool, guess_acc_bool)

                #Load probability if exists
                if filename in listdir('prob'):
                    with open('prob//'+filename, 'rb') as f:
                        tmpdata = load(f)
                    test_accs[modeltype_ind, :, model_num_ind, component_num_ind, :] = tmpdata['accs']
                    test_early[modeltype_ind, :, model_num_ind, component_num_ind, :] = tmpdata['earliness']

                #Otherwise, cross-validation
                else:
                    for fold_ind in range(len(train_inds)):
                        Xk = [X[ii] for ii in test_inds[fold_ind]]
                        Yk = [Ys[modeltype_ind][ii] for ii in test_inds[fold_ind]]
                        Tk = [T[ii] for ii in test_inds[fold_ind]]
                        class_weight = get_class_weight([Ys[modeltype_ind][ii][-1] for ii in train_inds[fold_ind]],
                            class_num_arr[modeltype_ind])
                        test_accs[modeltype_ind, fold_ind, model_num_ind, component_num_ind, :] ,\
                        test_early[modeltype_ind, fold_ind, model_num_ind, component_num_ind, :] = get_acc(model_foldername,
                                        modeltype_ind, model_split[modeltype_ind][fold_ind,:],
                                        fold_ind, k, Xk, Yk, Tk, thresh_arr, class_weight,
                                        class_num_arr[modeltype_ind], '', False,
                                        guess_bool, guess_acc_bool)
                        print('{}/{}'.format(cur_num, tot_num))
                        cur_num+=1

                    #Save data
                    my_data = {'accs': test_accs[modeltype_ind, :, model_num_ind, component_num_ind, :],
                            'earliness': test_early[modeltype_ind, :, model_num_ind, component_num_ind, :]}
                    # with open('prob//'+filename, 'wb') as output:
                    #     dump(my_data, output)
    print('_________________')
    print('Choosing Hyperparameters:')
    alpha = 1.0
    best_inds = [[[0], [0], [0]], [[1], [0], [3]]]

    for modeltype_ind in range(len(model_split)):

        #Average over classes and folds
        cur_test_accs = np.mean(test_accs[modeltype_ind, :, :, :, :], axis=0)
        cur_test_early = np.mean(test_early[modeltype_ind, :, :, :, :], axis=0)
        metric = alpha*cur_test_accs + (1-alpha)*cur_test_early
        print(np.std(test_accs[modeltype_ind, :, best_inds[modeltype_ind][0], best_inds[modeltype_ind][1], best_inds[modeltype_ind][2]]))
        #Choose best hyperparameters

        ind1, ind2, ind3 = best_inds[modeltype_ind]
        best_model_num = num_model_arr[ind1]
        best_component_num = num_component_arr[ind2]
        best_thresh = thresh_arr[ind3]


        print('')
        print('\t {} - All Best Model Number'.format(modeltype_name_arr[modeltype_ind]), best_model_num)
        print('\t {} - All Best Component Number'.format(modeltype_name_arr[modeltype_ind]), best_component_num)
        print('\t {} - All Best Thresholds'.format(modeltype_name_arr[modeltype_ind]), np.round(best_thresh, decimals=2))

        tmp_ind = 0
        print('\t {} - Best Model Number {}, Best Component Number {}, Best Threshold {:.5f}, '.format(
                    modeltype_name_arr[modeltype_ind], best_model_num[tmp_ind], best_component_num[tmp_ind], best_thresh[tmp_ind]))
        print('\t {} - For best parameters - Metric {:.5f}, Accuracy {:.5f}, Earliness {:.5f}'.format(
                    modeltype_name_arr[modeltype_ind], metric[ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]],
                    cur_test_accs[ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]],
                    cur_test_early[ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]]))
        # best_params.append((best_model_num[tmp_ind], best_component_num[tmp_ind], best_thresh[tmp_ind]))
        # best_inds.append((ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]))

    end = timer()
    print('Total Minutes - {:.4}'.format((end - start)/60))


if __name__ == '__main__':
    main()
