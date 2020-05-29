import numpy as np
from helper import choose_model, class_name_from_ind, get_metrics
from helper import get_prob, legend_from_ind
from plotting import plot_probability, plot_confusion_matrix
from multiprocessing import Pool

def propogate(params):
    x, y, t, models, num_classes, thresh_arr, class_weight, img_name, name, \
            cur_series, num_series, plot_graphs, mode_label, incr_arr = params

    step = 1.0

    num_points = x.shape[0]

    new_t = np.arange(0, np.ceil(t[-1]+step), step)
    pred_labels = np.zeros((incr_arr.shape[0], thresh_arr.shape[0])).astype('int')
    earliness = np.zeros((incr_arr.shape[0], thresh_arr.shape[0]))

    label = y[-1].astype('int')[0]

    avg_xs = []
    model_probs = []

    cur_model_ind = 0
    for ii, tt in enumerate(new_t):
        cur_inds = np.where((t >= tt-step) & (t < tt))[0]

        if cur_inds.shape[0] == 0:
            model_probs.append(np.zeros((1, num_classes)))
        else:
            model_ind = choose_model(t[cur_inds[-1]], models[0])
            if model_ind == cur_model_ind:
                avg_xs.append(np.mean(x[cur_inds,:], axis=0))

            else:
                if len(avg_xs) > 0:
                    tmp = get_prob(models[1][cur_model_ind], np.array(avg_xs), num_classes)
                    model_probs.append(tmp)
                cur_model_ind = model_ind
                avg_xs = [np.mean(x[cur_inds,:], axis=0)]

    if len(avg_xs) > 0:
        model_probs.append(get_prob(models[1][cur_model_ind], np.array(avg_xs), num_classes))

    model_probs = np.vstack(model_probs)
    model_probs[model_probs < 1e-6] = 1e-6

    class_weight[class_weight < 0.05] = 0.05
    class_weight = class_weight/np.sum(class_weight)

    for incr_ind, incr in enumerate(incr_arr):
        probs = np.zeros((new_t.shape[0], num_classes))
        probs[0,:] = class_weight

        for ii, tt in enumerate(new_t[1:]):
            if np.sum(model_probs[ii,:]) > 1e-6:
                probs[ii+1,:] = probs[ii,:] + incr*(model_probs[ii,:] - probs[ii,:])
                probs[ii+1,:] = probs[ii+1,:]/(np.sum(probs[ii+1,:]))
            else:
                probs[ii+1,:] = probs[ii,:]

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


            if ind_of_classification == -1:
                pred_label = mode_label

            #Only Guessing
            # pred_label = mode_label
            # ind_of_classification = 0

            pred_labels[incr_ind, thresh_ind] = pred_label

            if new_t[-1] < 1e-6:
                earliness[incr_ind, thresh_ind] = (new_t[-1] - new_t[ind_of_classification])/(new_t[-1] + 1e-6)
            else:
                earliness[incr_ind, thresh_ind] = (new_t[-1] - new_t[ind_of_classification])/(new_t[-1])

    return label, pred_labels, earliness

def run_models(X, Y, T, models, thresh_arr, incr_arr, class_weight, num_classes = 3,
               plot_graphs=False, plot_confusions=False, name='test',
               img_name = '', num_workers = 3, incr=0.05):

    res = []
    params = []
    mode_label = np.argmax(class_weight)
    for ii in range(len(X)):
        res.append(propogate((X[ii], Y[ii], T[ii], models, num_classes,
                    thresh_arr, class_weight, img_name, name, ii, len(X),
                    plot_graphs, mode_label, incr_arr)))

    acc, early_mats = get_metrics(res, thresh_arr, incr_arr, num_classes)

    return acc, early_mats

# if plot_graphs:
#     if ind_of_classification == -1:
#         count_text = "{:.0%} Threshold\nClassified at: end".format(
#             thresh)
#     else:
#         count_text = "{:.0%} Threshold\nClassified at: {} sec".format(
#                 thresh, new_t[ind_of_classification])
#
#     legend = legend_from_ind(num_classes)
#
#     title = "{}: {}/{}\n True Label: {}, Predicted Label: {}".format(name,
#              cur_series+1, num_series, class_name_from_ind(label, num_classes),
#              class_name_from_ind(pred_label, num_classes))
#
#     if label == pred_label:
#         result = 'Correct'
#     else:
#         result = 'Incorrect'
#
#     full_img_name = '{}_{:.4f}_Series_{:03d}.png'.format(img_name, thresh, cur_series)
#
#     plot_probability(new_t, probs, legend, title, result, count_text,
#             full_img_name)
