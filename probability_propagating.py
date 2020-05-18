import numpy as np
from helper import choose_model, class_name_from_ind, get_metrics
from helper import get_prob, legend_from_ind
from plotting import plot_probability, plot_confusion_matrix
from multiprocessing import Pool

def propogate(params):
    x, y, t, models, num_classes, thresh_arr, class_weight, img_name, name, \
            cur_series, num_series, plot_graphs, mode_label = params

    step = 1.0

    num_points = x.shape[0]

    class_weight[class_weight < 0.05] = 0.05
    class_weight = class_weight/np.sum(class_weight)
    prev_prob = np.ones_like(class_weight)/class_weight.shape[0]

    new_t = np.arange(0, np.ceil(t[-1]), step)
    probs = np.zeros((new_t.shape[0], num_classes))
    probs[0,:] = prev_prob

    # model_probs = np.zeros((new_t.shape[0], num_classes))
    # prev_t_ii = -1
    for ii, tt in enumerate(new_t[1:]):
        cur_inds = np.where((t <= tt) & (t >= tt-step))[0]

        if cur_inds.shape[0] == 0:
            probs[ii+1,:] = prev_prob
            continue

        model_ind = choose_model(t[cur_inds[-1]], models[0])
        avg_x = np.mean(x[cur_inds,:], axis=0).reshape(1, x[0,:].shape[0])
        model_prob = get_prob(models[1][model_ind], avg_x, num_classes)
        model_prob[model_prob < 1e-6] = 1e-6
        prev_prob[prev_prob < 1e-6] = 1e-6
        probs[ii+1,:] = model_prob*class_weight*prev_prob
        probs[ii+1,:] = probs[ii+1,:]/(np.sum(probs[ii+1,:]))
        # probs[ii+1,:] = prev_prob + 0.01*(ii - prev_t_ii)*(model_prob - prev_prob)
        # probs[ii+1,:] = probs[ii+1,:]/np.sum(probs[ii+1,:])
        # prev_t_ii = ii
        # probs[ii+1,:] = model_prob*prev_prob


        # if cur_series == 10:
        #     print(np.round(model_probs[ii,:], decimals=2), np.round(probs[ii,:], decimals=2))

        prev_prob = probs[ii,:]
    label = y[-1].astype('int')[0]

    pred_labels = np.empty_like(thresh_arr).astype('int')
    earliness = np.empty_like(thresh_arr)

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
            pred_label = mode_label

        pred_labels[thresh_ind] = pred_label
        earliness[thresh_ind] = (new_t[-1] - new_t[ind_of_classification])/(new_t[-1] + 1e-6)

    if cur_series%10 == 0:
        print('\t\t Probability {}/{}'.format(cur_series+1, num_series))

    return label, pred_labels, earliness

def run_models(X, Y, T, models, thresh_arr, class_weight, num_classes = 3,
               plot_graphs=False, plot_confusions=False, name='test',
               img_name = '', num_workers = 3, incr=0.05):

    res = []
    params = []
    mode_label = np.argmax(class_weight)
    for ii in range(len(X)):
        res.append(propogate((X[ii], Y[ii], T[ii], models, num_classes,
                    thresh_arr, class_weight, img_name, name, ii, len(X),
                    plot_graphs, mode_label)))

    # pool = Pool(processes=num_workers)
    # res = pool.map(propogate, params)

    num_thresh = thresh_arr.shape[0]
    conf_mats = np.zeros((num_thresh,num_classes,num_classes))

    all_res = np.zeros((num_thresh, len(X)))
    early_mats = np.zeros((num_thresh, len(res)))

    for ii, (label, pred_labels, earliness) in enumerate(res):
        for thresh_ind, pred_label, early in zip(range(num_thresh), pred_labels, earliness):
            conf_mats[thresh_ind, label, pred_label] += 1
            if label == pred_label:
                all_res[thresh_ind, ii] = 1
            early_mats[thresh_ind, ii] = early
    early_mats = np.mean(early_mats, axis=1)

    if plot_confusions:
        for thresh_ind, thresh in enumerate(thresh_arr):
            full_img_name = '{}_{:.4f}_Confusion.png'.format(img_name, thresh, ii)

            legend = legend_from_ind(num_classes)
            plot_confusion_matrix(conf_mats[thresh_ind, :, :], num_classes, legend, name,
                    full_img_name)

    acc = get_metrics(conf_mats, num_classes)
    #thresh_not_reached_arr, len(X)*np.ones_like(thresh_not_reached_arr)
    return acc, all_res, early_mats
