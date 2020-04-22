import numpy as np
from helper import choose_model, class_name_from_ind, get_metrics
from helper import get_prob, legend_from_ind
from plotting import plot_probability, plot_confusion_matrix
from multiprocessing import Pool

def propogate(params):
    x, y, t, models, num_classes, thresh_arr, class_weight, img_name, name, \
            cur_series, num_series, plot_graphs, incr = params

    step = 1.0

    num_points = x.shape[0]
    probs = np.zeros((num_points, num_classes))

    prev_prob = class_weight
    new_t = np.arange(step, np.ceil(t[-1]), step)
    probs = np.zeros((new_t.shape[0], num_classes))
    for ii, tt in enumerate(new_t):

        cur_inds = np.where((t <= tt) & (t >= tt-step))[0]

        if cur_inds.shape[0] == 0:
            probs[ii,:] = prev_prob
            continue

        model_ind = choose_model(t[cur_inds[-1]], models[0])
        avg_x = np.mean(x[cur_inds,:], axis=0).reshape(1, x[0,:].shape[0])
        model_prob = get_prob(models[1][model_ind], avg_x, num_classes)
        probs[ii,:] = prev_prob + incr*(model_prob - prev_prob)
        probs[ii,:] = probs[ii,:]/np.sum(probs[ii,:])
        #probs[ii,:] = model_prob/(prev_prob + 1e-6)
        #probs[ii,:] = probs[ii,:]/(np.sum(probs[ii,:]) + 1e-6)
        prev_prob = probs[ii,:]

    label = y[-1].astype('int')[0]

    pred_labels = np.empty_like(thresh_arr).astype('int')

    for thresh_ind, thresh in enumerate(thresh_arr):
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

        if plot_graphs:
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

            full_img_name = '{}_{:.4f}_Series_{:03d}.png'.format(img_name, thresh, cur_series)

            plot_probability(new_t, probs, legend, title, result, count_text,
                    full_img_name)

        if ind_of_classification == -1:
            pred_label = -1

        pred_labels[thresh_ind] = pred_label
    print('\t\t Fit {}/{}'.format(cur_series+1, num_series))

    return label, pred_labels

def run_models(X, Y, T, models, thresh_arr, class_weight, num_classes = 3,
               plot_graphs=False, plot_confusions=False, name='test',
               img_name = '', num_workers = 3, incr=0.05):


    if plot_graphs or plot_confusions:
        res = []
        for ii in range(len(X)):
            res.append(propogate((X[ii], Y[ii], T[ii], models, num_classes,
                    thresh_arr, class_weight, img_name, name, ii, len(X),
                    plot_graphs, incr)))
    else:
        res = []
        for ii in range(len(X)):
            res.append(propogate((X[ii], Y[ii], T[ii], models, num_classes,
                        thresh_arr, class_weight, img_name, name, ii, len(X),
                        plot_graphs, incr)))

        # param_vec = []
        # for ii in range(len(X)):
        #     param_vec.append((X[ii], Y[ii], T[ii], models, num_classes,
        #             thresh_arr, class_weight, img_name, name, ii, len(X),
        #             plot_graphs, incr))
        #
        # pool = Pool(processes=num_workers)
        # res = pool.map(propogate, param_vec)

    num_thresh = thresh_arr.shape[0]
    mode_label = np.argmax(class_weight)
    conf_mats = np.zeros((num_thresh,num_classes,num_classes))

    thresh_not_reached_arr = np.zeros((num_thresh))
    for label, pred_labels in res:

        thresh_reached = np.where(pred_labels >= 0)[0]
        thresh_not_reached = np.where(pred_labels < 0)[0]
        for thresh, pred_label in zip(thresh_reached, pred_labels[thresh_reached]):
            conf_mats[thresh, label, pred_label] += 1
        for thresh in thresh_not_reached:
            thresh_not_reached_arr[thresh] += 1
            conf_mats[thresh, label, mode_label] += 1

    if plot_confusions:
        for thresh_ind, thresh in zip(thresh_arr):
            full_img_name = '{}_{:.4f}_Confusion.png'.format(img_name, thres, ii)

            legend = legend_from_ind(num_classes)
            plot_confusion_matrix(conf_mats[thresh_ind, :, :], num_classes, legend, name,
                    full_img_name)

    acc = get_metrics(conf_mats, num_classes)

    return acc, thresh_not_reached_arr, len(X)*np.ones_like(thresh_not_reached_arr)
