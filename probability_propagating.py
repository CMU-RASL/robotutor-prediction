import numpy as np
from helper import choose_model, class_name_from_ind, get_metrics, get_prob
from plotting import plot_probability, plot_confusion_matrix

def propogate(x, t, models, num_classes, incr=0.05, step=1.0):

    num_points = x.shape[0]
    probs = np.zeros((num_points, num_classes))

    prev_prob = 1.0/num_classes*np.ones((num_classes))
    new_t = np.arange(step, np.ceil(t[-1]), step)
    probs = np.zeros((new_t.shape[0], num_classes))
    for ii, tt in enumerate(new_t):
        cur_inds = np.where((t < tt) & (t >= tt-step))[0]
        model_ind = choose_model(t[cur_inds[-1]], models[0])
        avg_x = np.mean(x[cur_inds,:], axis=0).reshape(1, x[0,:].shape[0])
        model_prob = get_prob(models[1][model_ind], avg_x, num_classes)
        probs[ii,:] = prev_prob + incr*(model_prob - prev_prob)
        probs[ii,:] = probs[ii,:]/np.sum(probs[ii,:])
        prev_prob = probs[ii,:]

    pred = np.argmax(probs, axis=1).astype('int')

    return probs, pred, new_t

def run_models(X, Y, T, models, thresh, num_classes = 3,
               plot_graphs=True, plot_confusions=True, name='test',
               img_name = ''):

    conf_mat = np.zeros((num_classes,num_classes))

    for ii in range(len(X)):
        probs, pred, new_T = propogate(X[ii], T[ii], models, num_classes)

        label = Y[ii][-1].astype('int')[0]

        if num_classes == 3:
            ind1, ind2 = np.where(probs[:,0] > thresh)[0], np.where(
                    probs[:,2] > thresh)[0]
            pred_label = 1
            ind_of_classification = -1
            if ind1.shape[0] > 0 and ind2.shape[0] > 0:
                if np.min(ind1) < np.min(ind2):
                    pred_label = 0
                    ind_of_classification = np.min(ind1)
                elif np.min(ind1) > np.min(ind2):
                    pred_label = 2
                    ind_of_classification = np.min(ind2)
                else:
                    pred_label = 1
            elif not ind1.shape[0] > 0 and ind2.shape[0] > 0:
                pred_label = 2
                ind_of_classification = np.min(ind2)
            elif ind1.shape[0] > 0 and not ind2.shape[0] > 0:
                pred_label = 0
                ind_of_classification = np.min(ind1)

            if ind_of_classification == -1:
                count_text = "{:.0%} Threshold\nClassified at: end".format(
                    thresh)
            else:
                count_text = "{:.0%} Threshold\nClassified at: {} sec".format(
                        thresh, new_T[ind_of_classification])
            legend = ['Negative', 'Neutral', 'Positive']
        else:
            ind1 = np.where(probs[:,1] > thresh)[0]
            pred_label = 0
            ind_of_classification = -1

            if ind1.shape[0] > 0:
                pred_label = 1
                ind_of_classification = np.min(ind1)

            if ind_of_classification == -1:
                count_text = "{:.0%} Threshold\nClassified at: end".format(
                    thresh)
            else:
                count_text = "{:.0%} Threshold\nClassified at: {} sec".format(
                        thresh, new_T[ind_of_classification])

            legend = ['Completed', 'Bailed']

        title = "{}: {}/{}\n True Label: {}, Predicted Label: {}".format(name,
                 ii+1, len(X), class_name_from_ind(label, num_classes),
                 class_name_from_ind(pred_label, num_classes))

        if label == pred_label:
            result = 'Correct'
        else:
            result = 'Incorrect'

        full_img_name = '{}_Series_{:03d}.png'.format(img_name, ii)
        plot_probability(new_T, probs, legend, title, result, count_text, full_img_name, plot_graphs)

        conf_mat[label, pred_label] += 1

    full_img_name = '{}_Confusion.png'.format(img_name, ii)
    plot_confusion_matrix(conf_mat, num_classes, legend, name, full_img_name, plot_confusions)

    fpr, tpr, acc = get_metrics(conf_mat, num_classes)

    return fpr, tpr, acc
