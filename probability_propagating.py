import numpy as np
from helper import choose_model, class_name_from_ind, get_metrics
from helper import get_prob, legend_from_ind
from plotting import plot_probability, plot_confusion_matrix
from multiprocessing import Pool

def propogate(params):
    x, y, t, models, num_classes, thresh, img_name, name, \
            cur_series, num_series, plot_graphs = params
    incr = 0.05
    step = 1.0

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

    label = y[-1].astype('int')[0]

    inds = [np.where(probs[:,class_num] > thresh)[0]
            for class_num in range(num_classes)]
    num_inds = [inds[class_num].shape[0] for class_num in range(num_classes)]
    max_inds = [probs.shape[0]+1 for class_num in range(num_classes)]
    for class_num in range(num_classes):
        if num_inds[class_num] > 0:
            max_inds[class_num] = np.min(inds[class_num])

    pred_label = np.argmin(max_inds)
    if num_inds[pred_label] == 0:
        pred_label = num_classes - 2
        ind_of_classification = -1
    else:
        ind_of_classification = max_inds[pred_label]

    if ind_of_classification == -1:
        count_text = "{:.0%} Threshold\nClassified at: end".format(
            thresh)
    else:
        count_text = "{:.0%} Threshold\nClassified at: {} sec".format(
                thresh, new_t[ind_of_classification])

    legend = legend_from_ind(num_classes)

    title = "{}: {}/{}\n True Label: {}, Predicted Label: {}".format(name,
             cur_series+1, num_series, class_name_from_ind(label, num_classes),
             class_name_from_ind(pred_label, num_classes))

    if label == pred_label:
        result = 'Correct'
    else:
        result = 'Incorrect'

    full_img_name = '{}_Series_{:03d}.png'.format(img_name, cur_series)

    if plot_graphs:
        plot_probability(new_t, probs, legend, title, result, count_text,
                full_img_name)

    return label, pred_label

def run_models(X, Y, T, models, thresh, num_classes = 3,
               plot_graphs=True, plot_confusions=True, name='test',
               img_name = '', num_workers = 5):

    conf_mat = np.zeros((num_classes,num_classes))
    for ii in range(len(X)):
        label, pred_label = propogate((X[ii], Y[ii], T[ii], models, num_classes,
                thresh, img_name, name, ii, len(X), plot_graphs))
        conf_mat[label, pred_label] += 1

        # if ii % 20 == 0:
        #     print('\t\t\t {}: {}/{}'.format(name,ii+1, len(X)))

    full_img_name = '{}_Confusion.png'.format(img_name, ii)

    legend = legend_from_ind(num_classes)
    if plot_confusions:
        plot_confusion_matrix(conf_mat, num_classes, legend, name,
                full_img_name)

    fpr, tpr, acc = get_metrics(conf_mat, num_classes)

    return fpr, tpr, acc
