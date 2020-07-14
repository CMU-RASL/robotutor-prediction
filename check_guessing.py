import numpy as np
from os import mkdir, listdir
from model_training import get_training_split, create_all_models, get_acc
from helper import remove_readonly, filter_data, get_class_weight
from shutil import rmtree
from pickle import dump, load
from timeit import default_timer as timer

def main(k=10,data_name='dataset2_back', plot_bool=False,feature_set='all',
    col_to_remove='None',alpha=0.7, beta=0.2, casenum=2, guess_bool = False, guess_acc_bool = True):

    start = timer()
    print('_________________')
    print('Setting Parameters:')

    #Hyperparameter arrays
    aa = np.arange(0.55, 1.0, 0.05)
    bb = np.arange(-0.3, 0.3, 0.05)

    A, B = np.meshgrid(aa, bb)
    A = A.flatten()
    B = B.flatten()
    num_thresh = A.shape[0]

    num_model_arr = np.arange(1, 7).astype('int')
    num_model = num_model_arr.shape[0]

    num_component_arr = np.arange(1, 7).astype('int')
    num_components = num_component_arr.shape[0]

    str_params = 'thresh_{:.4}_{:.4}_{:.4}_{:.4}_{:.4}_{:.4}_casenum_{}'.format(
        np.min(A), np.max(A), A[-1]-A[0], np.min(B), np.max(B), B[-1]-B[0], casenum)

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
    train_inds, test_inds = get_training_split(len(X), k)

    # test_filename = test_name + '.pkl'
    # #Load data
    # with open(test_filename, 'rb') as f:
    #     data = load(f)
    #
    # #Filter data
    # X_test, Y1_test, Y2_test, T_test, feat_names_test = data['X'], data['Y1'], data['Y2'], \
    #                             data['T'], data['feat_names']
    # X_test, Y1_test, Y2_test, T_test, feat_names_test = filter_data(X_test, Y1_test, Y2_test, T_test, feat_names_test,
    #                             feature_set, col_to_remove)
    # Ys_test = [Y1_test, Y2_test]


    print('_________________')
    print('Cross-Validation:')

    test_accs = np.zeros((2, k, num_model, num_components, num_thresh))
    test_early = np.zeros((2, k, num_model, num_components, num_thresh))
    test_thresh_met = np.zeros((2, k, num_model, num_components, num_thresh))
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

                filename = 'prob_data_{}_folds_{}_modeltype_{}_modelnum_{}_componentnum_{}_{}_features_{}_casenum_{}.pkl'.format(
                            data_name, k, modeltype_ind, model_num, component_num, str_params, feature_set, casenum)

                #Load probability if exists
                if filename in listdir('prob'):
                    with open('prob//'+filename, 'rb') as f:
                        tmpdata = load(f)
                    test_accs[modeltype_ind, :, model_num_ind, component_num_ind, :] = tmpdata['accs']
                    test_early[modeltype_ind, :, model_num_ind, component_num_ind, :] = tmpdata['earliness']
                    test_thresh_met[modeltype_ind, :, model_num_ind, component_num_ind, :] = tmpdata['thresh_met']

                #Otherwise, cross-validation
                else:
                    for fold_ind in range(len(train_inds)):
                        Xk = [X[ii] for ii in test_inds[fold_ind]]
                        Yk = [Ys[modeltype_ind][ii] for ii in test_inds[fold_ind]]
                        Tk = [T[ii] for ii in test_inds[fold_ind]]
                        class_weight = get_class_weight([Ys[modeltype_ind][ii][-1] for ii in train_inds[fold_ind]],
                            class_num_arr[modeltype_ind])
                        test_accs[modeltype_ind, fold_ind, model_num_ind, component_num_ind, :] ,\
                        test_early[modeltype_ind, fold_ind, model_num_ind, component_num_ind, :], \
                        test_thresh_met[modeltype_ind, fold_ind, model_num_ind, component_num_ind, :] = get_acc(model_foldername,
                                        modeltype_ind, model_split[modeltype_ind][fold_ind,:],
                                        fold_ind, k, Xk, Yk, Tk, A, B, class_weight,
                                        class_num_arr[modeltype_ind], '', False,
                                        guess_bool, guess_acc_bool)
                        print('{}/{}'.format(cur_num, tot_num))
                        cur_num+=1

                    #Save data
                    my_data = {'accs': test_accs[modeltype_ind, :, model_num_ind, component_num_ind, :],
                            'earliness': test_early[modeltype_ind, :, model_num_ind, component_num_ind, :],
                            'thresh_met': test_thresh_met[modeltype_ind, :, model_num_ind, component_num_ind, :]}
                    with open('prob//'+filename, 'wb') as output:
                        dump(my_data, output)

    print('_________________')
    print('Choosing Hyperparameters:')
    best_params = []
    best_vals = []
    for modeltype_ind in range(len(model_split)):

        #Average over classes and folds
        cur_test_accs = np.mean(test_accs[modeltype_ind, :, :, :, :], axis=0)
        cur_test_early = np.mean(test_early[modeltype_ind, :, :, :, :], axis=0)
        cur_thresh_met = np.mean(test_thresh_met[modeltype_ind, :, :, :, :], axis=0)
        metric = alpha*cur_test_accs + beta*cur_test_early + (1 - alpha - beta)*cur_thresh_met

        #Choose best hyperparameters
        max_metric = np.max(metric)
        ind1, ind2, ind3 = np.where(metric == max_metric)
        best_model_num = num_model_arr[ind1]
        best_component_num = num_component_arr[ind2]
        best_A = A[ind3]
        best_B = B[ind3]
        print('')
        print('\t {} - All Best Model Number'.format(modeltype_name_arr[modeltype_ind]), best_model_num)
        print('\t {} - All Best Component Number'.format(modeltype_name_arr[modeltype_ind]), best_component_num)
        print('\t {} - All Best A'.format(modeltype_name_arr[modeltype_ind]), np.round(best_A, decimals=3))
        print('\t {} - All Best B'.format(modeltype_name_arr[modeltype_ind]), np.round(best_B, decimals=3))

        tmp_ind = 0
        all_test_accs = test_accs[modeltype_ind, :, ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]]
        all_test_early = test_early[modeltype_ind, :, ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]]
        all_thresh_met = test_thresh_met[modeltype_ind, :, ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]]
        all_metric = alpha*all_test_accs + (beta)*all_test_early + (1 - alpha - beta)*all_thresh_met
        acc_stddev = np.std(all_test_accs)
        early_stddev = np.std(all_test_early)
        metric_stddev = np.std(all_metric)
        thresh_met_stddev = np.std(all_thresh_met)


        print('\t {} - Best Model Number {}, Best Component Number {}, Best A {:.5f}, Best B {:.5f}'.format(
                    modeltype_name_arr[modeltype_ind], best_model_num[tmp_ind], best_component_num[tmp_ind], best_A[tmp_ind], best_B[tmp_ind]))
        print('\t {} - For best parameters - Metric {:.3%} ({:.4%}), Accuracy {:.3%} ({:.4%}), Earliness {:.3%} ({:.4%}), Thresh Met {:.3%} ({:.4%})'.format(
                    modeltype_name_arr[modeltype_ind], max_metric, metric_stddev, cur_test_accs[ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]],
                    acc_stddev, cur_test_early[ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]], early_stddev,
                    cur_thresh_met[ind1[tmp_ind], ind2[tmp_ind], ind3[tmp_ind]], thresh_met_stddev))
        best_params.append((best_model_num[tmp_ind], best_component_num[tmp_ind], best_A[tmp_ind], best_B[tmp_ind]))
        best_vals.append((all_metric, all_test_accs, all_test_early))


    my_data = {'A_arr': A, 'B_arr': B, 'num_model_arr': num_model_arr,
        'num_component_arr': num_component_arr,
        'feat_names': feat_names, 'test_accs': test_accs,
        'test_early': test_early, 'best_vals': best_vals, 'best_params': best_params, 'thresh_met': test_thresh_met}

    result_filename = result_filename = 'result//result_alpha_{}_beta_{}_features_{}_casenum_{}.pkl'.format(alpha, beta, feature_set, casenum)
    with open(result_filename, 'wb') as output:
        dump(my_data, output)

    end = timer()
    print('Total Minutes - {:.4}, Saved to - {}'.format((end - start)/60, result_filename))


if __name__ == '__main__':
    main(feature_set='all', alpha = 1.0, beta = 0.0, data_name='dataset2', casenum=1, guess_bool=True, guess_acc_bool=True)
    # main(feature_set='all', alpha=1.0, data_name='dataset2', casenum=2, guess_bool=False, guess_acc_bool=True)
    main(feature_set='all', alpha = 1.0, beta = 0.0, data_name='dataset2', casenum=4, guess_bool=False, guess_acc_bool=False)

    # main(feature_set='all', alpha=1.0, data_name='dataset2', casenum=3, guess_bool=False, guess_acc_bool=False)
