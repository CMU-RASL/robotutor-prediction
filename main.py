import numpy as np
import os
from plotting import plot_accuracies, plot_rocs
from model_training import fit_model, load_model, get_training_split
from probability_propagating import run_models

def main():
    model_split = np.array([[0, 30, 100, 300, 450, -1], 
                            [0, 10, 20, 50, 200, -1]])
    num_thresh = 11
    k = 4
    thresh_arr = np.linspace(0, 1, num_thresh).astype('float')
    thresh_arr = [0.8]
    data = np.load('all_data.npz', allow_pickle = True)
    X, Y1, Y2, T = data['X'], data['Y1'], data['Y2'], data['T']
    Ys = [Y1, Y2]
    
    print('Get Training Data\n')
    train_inds, test_inds = get_training_split(X, Y1, Y2, T, k)
    
    filename = 'train_test_split.txt'
    f = open(filename, 'w+')
    for ii, test_ind in enumerate(test_inds):
        name = '-'.join(map(str, test_ind))
        f.write('{:02} \t {}\n'.format(ii, name))
    f.close()
    
    train_fprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    train_tprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    train_accs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_fprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_tprs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    test_accs = [np.empty((k, num_thresh, 3)), np.empty((k, num_thresh, 1))]
    class_num_arr = [3, 2]
    for model_ind in range(1, len(model_split)):
        print('Model {}/{}'.format(model_ind+1, 2))
        for batch_ind in range(len(train_inds)):
            cur_model_arr = [[],[]]
            for model_start_ind in range(len(model_split[model_ind])-1):
                start_val = model_split[model_ind][model_start_ind]
                end_val = model_split[model_ind][model_start_ind+1]
                model_name = 'Model_{}_Start_{}_End_{}_Batch_{}.joblib'.format(model_ind, 
                                    start_val, end_val, batch_ind)

                if not model_name in os.listdir('models'):
                    model = fit_model(X[train_inds[batch_ind]], 
                              Ys[model_ind][train_inds[batch_ind]],
                              T[train_inds[batch_ind]],
                              start_val, end_val, model_name, 
                              num_classes = class_num_arr[model_ind])
                else:
                    model = load_model(model_name)
                cur_model_arr[0].append((start_val, end_val))
                cur_model_arr[1].append(model)
            print('\t Loaded Models for Batch {}/{}'.format(batch_ind+1, k))
            
            for thresh_ind, thresh in enumerate(thresh_arr):
                fpr, tpr, acc = run_models(X[train_inds[batch_ind]], 
                                     Ys[model_ind][train_inds[batch_ind]], 
                                     T[train_inds[batch_ind]], cur_model_arr,
                                     thresh, num_classes = 
                                     class_num_arr[model_ind], 
                                     plot_graphs = False, 
                                     plot_confusions=False, name = 'train')
                train_fprs[model_ind][batch_ind, thresh_ind, :] = fpr
                train_tprs[model_ind][batch_ind, thresh_ind, :] = tpr
                train_accs[model_ind][batch_ind, thresh_ind, :] = acc
                
                fpr, tpr, acc = run_models(X[test_inds[batch_ind]], 
                                     Ys[model_ind][test_inds[batch_ind]], 
                                     T[test_inds[batch_ind]], cur_model_arr,
                                     thresh, num_classes = 
                                     class_num_arr[model_ind], 
                                     plot_graphs = True, 
                                     plot_confusions=True, name = 'test')
                test_fprs[model_ind][batch_ind, thresh_ind, :] = fpr
                test_tprs[model_ind][batch_ind, thresh_ind, :] = tpr
                test_accs[model_ind][batch_ind, thresh_ind, :] = acc
                print('\t\t Batch {}/{}, Thresh {}/{}'.format(batch_ind+1,
                      k,thresh_ind+1,num_thresh))
                return
    for model_ind in range(len(model_split)):
        plot_accuracies(train_accs[model_ind], thresh_arr, 'train')
        plot_accuracies(test_accs[model_ind], thresh_arr, 'test')
        plot_rocs(train_fprs[model_ind], train_tprs[model_ind], thresh_arr, 'train')
        plot_rocs(test_fprs[model_ind], test_tprs[model_ind], thresh_arr, 'test')
    
if __name__ == '__main__':
    main()

