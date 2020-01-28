import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from joblib import dump, load

def class_name_from_ind(ind, num_classes):
    if num_classes == 2:
        if ind == 0:
            return 'Completed'
        else:
            return 'Bailed'
    else:
        if ind == 0:
            return 'Negative'
        elif ind == 1:
            return 'Neutral'
        else:
            return 'Positive'

def get_unique_classes(Y1, Y2, split_name, print_flag=False):
    
    tot = np.empty(len(Y1))
    
    for ii, yi in enumerate(Y1):
        tot[ii] = int(yi[-1])
    
    perc1 = [np.sum(tot==0)/len(Y1), np.sum(tot==1)/len(Y1), 
            np.sum(tot==2)/len(Y1)]
    
    perc2 = [np.sum(tot==0)/len(Y2), np.sum(tot==1)/len(Y2)]
    
    if print_flag:
        print("{} split: {} trajs".format(split_name, len(tot)))
        print("{:.2%} Negative, {:.2%} Neutral, {:.2%} Positive".format(
              perc1[0], perc1[1], perc1[2]))
        print("{:.2%} Completed, {:.2%} Bailed".format(perc2[0], perc2[1]))
        print('')
    
    return perc1, perc2

def get_training_data(X, Y1, Y2, T, train_split, random = False):
    
    num_series = len(X)
    train_size = np.floor(num_series*train_split).astype('int')
    
    test_perc1 = [0.0, 0.0, 0.0]
    train_perc1 = [0.0,0.0,0.0]
    test_perc2 = [0.0, 0.0]
    train_perc2 = [0.0, 0.0]
    
    if not random:
        if train_size == 30: #0.8
            print('Using cached indices for 0.8 training')
            inds = [3, 36, 21, 14, 35,  6,  4,  8, 16, 15, 24,  2,  9, 31, 
                    37, 18, 26, 13,  5, 11, 17, 28,  0,  1, 34, 32, 30, 25, 
                    12, 20, 33, 19, 10,  7, 29, 27, 23, 22]
        elif train_size == 28: #0.75
            print('Using cached indices for 0.75 training')
            inds = [ 7, 33, 10,  3, 19, 15, 34,  8, 24,  0, 28, 25,  5, 13, 
                    18,29, 30, 36,  6, 16,  4, 17, 21, 37, 32, 14,  1, 31, 
                    35, 27, 23,  9, 11, 12, 26,  2, 22, 20]
        else:
            print('Did not save indices for this training percentage')
            
        train_X, train_Y1, train_Y2, train_T = X[inds[:train_size]], Y1[inds[:train_size]], Y2[inds[:train_size]], T[inds[:train_size]]
        
        test_X, test_Y1, test_Y2, test_T = X[inds[train_size:]], Y1[inds[train_size:]], Y2[inds[train_size:]], T[inds[train_size:]]
    else:
        while np.min(test_perc1) < 0.08 or np.min(train_perc1) < 0.08:
            inds = np.arange(len(X))
            np.random.shuffle(inds)
            
            train_X, train_Y1, train_Y2, train_T = X[inds[:train_size]], Y1[inds[:train_size]], Y2[inds[:train_size]], T[inds[:train_size]]
            
            test_X, test_Y1, test_Y2, test_T = X[inds[train_size:]], Y1[inds[train_size:]], Y2[inds[train_size:]], T[inds[train_size:]]
        
            test_perc1, test_perc2 = get_unique_classes(test_Y1, test_Y2, 'test')
            train_perc2, train_perc2 = get_unique_classes(train_Y1, train_Y2, 'train')
    
    all_perc1, all_perc2 = get_unique_classes(Y1, Y2, 'all', True)
    train_perc1, train_perc2 = get_unique_classes(train_Y1, train_Y2, 'train', True)
    test_perc1, test_perc2 = get_unique_classes(test_Y1, test_Y2, 'test', True)
    
    return train_X, train_Y1, train_Y2, train_T, test_X, test_Y1, test_Y2, test_T

def plot_probability(t, prob, legend, title, result, count_text):

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(legend) == 2:
        colors = ['b', 'm']
    else:
        colors = ['r', 'y', 'g']
        
    for ii, label in enumerate(legend):
        plt.plot(t, prob[:,ii], colors[ii], label=label)

    plt.title(title, fontsize=20)
    plt.legend(fontsize=18, bbox_to_anchor=(1, 1))
    plt.text(1.05, 0.2, count_text, fontsize=18, transform=ax.transAxes, 
             bbox=dict(boxstyle="round", fc="w"))
    if result == 'Correct':
        plt.text(1.05, 0.0, result, fontsize=20, color="b", 
                 transform=ax.transAxes, bbox=dict(boxstyle="round", 
                                                   alpha=0.2, fc="k"))
    else:
        plt.text(1.05, 0.0, result, fontsize=20, color="m", 
                 transform=ax.transAxes, bbox=dict(boxstyle="round", 
                                                   alpha=0.2, fc="k"))
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_mat, num_classes, legend, name):
    norm_conf_mat = conf_mat / np.sum(conf_mat)
    plt.figure(figsize=(9, 5))
    plt.imshow(conf_mat, cmap='Blues')
    acc = np.trace(conf_mat) / np.sum(conf_mat)
    plt.title('{} Confusion Matrix: Accuracy: {:.2%}'.format(name, acc), fontsize=24)
    plt.xticks(np.arange(num_classes), legend, fontsize=16)
    plt.yticks(np.arange(num_classes), legend, fontsize=16)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    cmap = plt.cm.get_cmap('Greys')
    for ii in range(num_classes):
        for jj in range(num_classes):
            plt.text(jj, ii, str(np.round(conf_mat[ii,jj], decimals=2)),
                     ha='center', color=cmap(1-norm_conf_mat[ii,jj]), 
                     fontsize=24)
    plt.tight_layout()

def choose_model(tt, model_split):
    for ii in range(len(model_split)-1):
        if tt >= model_split[ii] and tt < model_split[ii+1]:
            return ii
    return len(model_split)-2

def propogate(x, t, models, model_split, num_classes, incr=0.05):
    
    num_points = x.shape[0]
    probs = np.zeros((num_points, num_classes))
    
    prev_prob = 1.0/num_classes*np.ones((num_classes))
    for ii in range(num_points):
        model_ind = choose_model(t[ii], model_split)
        model_prob = models[model_ind].predict_proba(x[ii,:].reshape(1, x[ii,:].shape[0]))
        probs[ii,:] = prev_prob + incr*(model_prob - probs[ii,:])
        probs[ii,:] = probs[ii,:]/np.sum(probs[ii,:])
        prev_prob = probs[ii,:]
    
    pred = np.argmax(probs, axis=1).astype('int')
    
    return probs, pred

def run_models(X, Y, T, models, model_split, num_classes = 3, plot_graphs=True, name='test'):
    
    conf_mat = np.zeros((num_classes,num_classes))
    
    for ii in range(len(X)):
        probs, pred = propogate(X[ii], T[ii], models, model_split, num_classes)
        
        label = Y[ii][-1].astype('int')[0]
        
        
        if num_classes == 3:
            neg_count = np.where(pred==0)[0].shape[0]/pred.shape[0]
            neu_count = np.where(pred==1)[0].shape[0]/pred.shape[0]
            pos_count = np.where(pred==2)[0].shape[0]/pred.shape[0]          
    
            count_text = "{:.1%} Negative\n{:.1%} Neutral\n{:.1%} Positive".format(
                    neg_count, neu_count, pos_count)
            
            legend = ['Negative', 'Neutral', 'Positive']
        else:
            completed_count = np.where(pred==0)[0].shape[0]/pred.shape[0]
            bail_count = np.where(pred==1)[0].shape[0]/pred.shape[0]
            
            count_text = "{:.1%} Completed\n{:.1%} Bailed".format(
                    completed_count, bail_count)
            
            legend = ['Completed', 'Bailed']
    
        title = "{}: {}/{}\n True Label: {}, Predicted Label: {}".format(name, 
                 ii+1, len(X), class_name_from_ind(label, num_classes), 
                 class_name_from_ind(pred[-1], num_classes)) 

        if label == pred[-1]:
            result = 'Correct'
        else:
            result = 'Incorrect'
        
        if plot_graphs:
            plot_probability(T[ii], probs, legend, title, result, count_text)
        else:
            print(title)
    
        conf_mat[label, pred[-1]] += 1
        
    plot_confusion_matrix(conf_mat, num_classes, legend, name)

def fit_model(train_X, train_Y, train_T, start_val, end_val):
    
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
    
    flat_train_X = np.vstack(X)
    flat_train_Y = np.vstack(Y).flatten()
    unique_elements, counts_elements = np.unique(flat_train_Y, 
                                                 return_counts=True)
    
    weights = counts_elements / flat_train_Y.shape[0]
    class_weight= {}
    for label, weight in zip(unique_elements, weights):
        class_weight[label] = weight
        
    model = SVC(probability=True, gamma='auto', class_weight=class_weight)
    model.fit(flat_train_X, flat_train_Y)
    
    return model

def create_models(train_X, train_Y, train_T, folder,  
                  model_split = [0, 10, 20, 30, 60, 90, -1]):
    
    for ii in range(len(model_split)-1):
        print('Fitting model from {} to {} seconds'.format(model_split[ii], 
              model_split[ii+1]))
        
        model = fit_model(train_X, train_Y, train_T, model_split[ii],
                          model_split[ii+1])
        
        filename = '{}/{}_{}_model.joblib'.format(folder, model_split[ii], 
                           model_split[ii+1])
        
        dump(model, filename)
        
        print('Saved model to {}'.format(filename))

def load_models(model_split = [0, 10, 20, 30, 60, 90, -1], folders = ['models1', 'models2']):
    
    models = [[],[]]
    for ii in range(len(model_split)-1):
        for jj, folder in enumerate(folders):
            filename = '{}/{}_{}_model.joblib'.format(folder, model_split[ii], 
                               model_split[ii+1])
            model = load(filename)
            
            models[jj].append(model)
    
    return models

def main():
    train_split = 0.8
    model_split = [0, 10, 20, 30, 60, 90, -1]
    
    data = np.load('all_data.npz', allow_pickle = True)
    X, Y1, Y2, T = data['X'], data['Y1'], data['Y2'], data['T']
    
    train_X, train_Y1, train_Y2, train_T, test_X, test_Y1, test_Y2, test_T = get_training_data(X, Y1, Y2, T, 
                                                         train_split, random = False)
    
#    create_models(train_X, train_Y1, train_T, 'models1', model_split)
#    create_models(train_X, train_Y2, train_T, 'models2', model_split)
    models = load_models(model_split, ['models1', 'models2'])
    model1 = models[0]
    model2 = models[1]
    
#    run_models(train_X, train_Y1, train_T, model1, model_split, num_classes = 3, plot_graphs = False, name = 'train')
#    run_models(train_X, train_Y2, train_T, model2, model_split, num_classes = 2, plot_graphs = False, name = 'train')

#    run_models(test_X, test_Y1, test_T, model1, model_split, num_classes = 3, plot_graphs = True, name = 'test')
#    run_models(test_X, test_Y2, test_T, model2, model_split, num_classes = 2, plot_graphs = True, name = 'test')
    
if __name__ == '__main__':
    main()

