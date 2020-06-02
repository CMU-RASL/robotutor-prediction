import numpy as np
from helper import choose_model, class_name_from_ind, get_metrics
from helper import get_prob, legend_from_ind
from plotting import plot_probability, plot_confusion_matrix
from multiprocessing import Pool
from scipy.special import expit

def propogate(params):
    x, y, t, models, num_classes, A, B, class_weight, img_name, name, \
            cur_series, num_series, plot_graphs, mode_label, guess_bool = params

    step = 1.0

    num_points = x.shape[0]

    class_weight[class_weight < 0.05] = 0.05
    class_weight = class_weight/np.sum(class_weight)
    prev_prob = np.ones_like(class_weight)/class_weight.shape[0]

    new_t = np.arange(0, np.ceil(t[-1]), step)
    probs = np.zeros((new_t.shape[0], num_classes))
    probs[0,:] = prev_prob

    for ii, tt in enumerate(new_t[1:]):
        cur_inds = np.where((t <= tt) & (t >= tt-step))[0]

        if cur_inds.shape[0] == 0:
            probs[ii+1,:] = prev_prob
            # if cur_series == 9:
                # print(np.zeros((1,17)))
            continue

        model_ind = choose_model(t[cur_inds[-1]], models[0])
        avg_x = np.mean(x[cur_inds,:], axis=0).reshape(1, x[0,:].shape[0])

        model_prob = get_prob(models[1][model_ind], avg_x, num_classes)
        model_prob[model_prob < 1e-6] = 1e-6
        prev_prob[prev_prob < 1e-6] = 1e-6

        probs[ii+1,:] = model_prob*class_weight*prev_prob
        probs[ii+1,:] = probs[ii+1,:]/(np.sum(probs[ii+1,:]))

        prev_prob = probs[ii,:]

        # if cur_series == 9:
        #     print(avg_x)
    # print('Done')
    label = y[-1].astype('int')[0]

    pred_labels = np.empty_like(A).astype('int')
    earliness = np.empty_like(A)

    for thresh_ind, (aa, bb) in enumerate(zip(A, B)):
        if guess_bool:
            pred_label = mode_label
            ind_of_classification = 0
        else:
            thresh = aa*np.exp(bb*new_t)
            thresh[thresh>1.0] = 1.0
            thresh[thresh<0.3] = 0.3
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

            #Threshold not met
            if ind_of_classification == -1:
                pred_label = mode_label
        if plot_graphs:
            if ind_of_classification == -1:
                count_text = "{:.0%}, {:.1%} Threshold\nClassified at: end".format(
                    aa, bb)
            else:
                count_text = "{:.0%}, {:.1%} Threshold\nClassified at: {} sec".format(
                        aa, bb, new_t[ind_of_classification])

            legend = legend_from_ind(num_classes)

            title = "{}: {}/{}\n True Label: {}, Predicted Label: {}".format(name,
                     cur_series+1, num_series, class_name_from_ind(label, num_classes),
                     class_name_from_ind(pred_label, num_classes))

            if label == pred_label:
                result = 'Correct'
            else:
                result = 'Incorrect'

            full_img_name = '{}_{:.4}_{:.4}_Series_{:03d}.png'.format(img_name, aa, bb, cur_series+1)

            plot_probability(new_t, probs, legend, title, result, count_text,
                    full_img_name, thresh)

        pred_labels[thresh_ind] = pred_label
        if new_t[-1] < 1e-6:
            earliness[thresh_ind] = (new_t[-1] - new_t[ind_of_classification])/(new_t[-1] + 1e-6)
        else:
            earliness[thresh_ind] = (new_t[-1] - new_t[ind_of_classification])/(new_t[-1])

    return label, pred_labels, earliness

def run_models(X, Y, T, models, A, B, class_weight, num_classes = 3,
               plot_graphs=False, plot_confusions=False, name='test',
               img_name = '', num_workers = 3, incr=0.05, guess_bool=True,
               guess_acc_bool = True):

    res = []
    params = []
    if guess_acc_bool:
        mode_label = np.argmax(class_weight)
    else:
        mode_label = 10

    for ii in range(len(X)):
        res.append(propogate((X[ii], Y[ii], T[ii], models, num_classes,
                    A, B, class_weight, img_name, name, ii, len(X),
                    plot_graphs, mode_label, guess_bool)))

    acc, early_mats = get_metrics(res, A.shape[0], num_classes)

    return acc, early_mats
