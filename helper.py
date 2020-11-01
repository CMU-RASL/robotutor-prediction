import numpy as np
import os
import cv2
import stat
from sklearn.metrics import f1_score

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def legend_from_ind(num_classes):
    if num_classes == 2:
        legend = ['Completed', 'Bailed']
    else:
        legend = ['Negative', 'Neutral', 'Positive']
    return legend


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

def choose_model(tt, model_split):
    for ii in range(len(model_split)):
        if tt >= model_split[ii][0] and tt < model_split[ii][1]:
            return ii
    return len(model_split)-1

def get_prob(models, x, num_classes):
    pred = np.zeros((len(models)))
    for ii, model in enumerate(models):
        pred[ii] = model.score_samples(x)[0]
    pred = np.exp(pred)
    # pred = pred / np.sum(pred)
    return pred

def get_metrics(res, num_thresh, num_classes):

    conf_mats = np.zeros((num_thresh,num_classes,num_classes))

    early_mats = np.zeros((num_thresh, len(res)))
    thresh_met_mats = np.zeros((num_thresh, len(res)))
    y_true = np.zeros((num_thresh, len(res)))
    y_pred = np.zeros((num_thresh, len(res)))

    for ii, (label, pred_labels, earliness, thresh_met) in enumerate(res):
        for thresh_ind, pred_label, early, met in zip(range(num_thresh), pred_labels, earliness, thresh_met):

            if pred_label < 5:
                conf_mats[thresh_ind, label, pred_label] += 1
                early_mats[thresh_ind, ii] = early
                y_true[thresh_ind, ii] = label
                y_pred[thresh_ind, ii] = pred_label
                thresh_met_mats[thresh_ind, ii] += met


    early_mats = np.mean(early_mats, axis=1)
    thresh_met_mats = np.sum(thresh_met_mats, axis=1)/len(res)
    acc = np.empty((conf_mats.shape[0], 1))

    for ind in range(conf_mats.shape[0]):
        conf_mat = conf_mats[ind, :, :]

        if np.sum(conf_mat) < 1e-6:
            acc[ind, 0] = np.trace(conf_mat)/(np.sum(conf_mat) + 1e-6)
        else:
            acc[ind, 0] = np.trace(conf_mat)/(np.sum(conf_mat))
        acc[ind,0] = f1_score(y_true[ind,:], y_pred[ind,:], average='weighted')
    return acc, early_mats, thresh_met_mats

def picture_side(folder_name, video_filename):

    cap = cv2.VideoCapture(folder_name + '/videos/' + video_filename)

    sides = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total', length)
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        height, width, channels = frame.shape

        left_wind = frame[height//2 - 200 : height//2 + 200,
                          width // 4 - 200 : width // 4 + 200,:]

        right_wind = frame[height//2 - 200 : height//2 + 200,
                           3 * width // 4 - 200 : 3 * width // 4 + 200,:]

        colors, count = np.unique(left_wind.reshape(-1,left_wind.shape[-1]),
                                  axis=0, return_counts=True)
        left = colors[count.argmax()]

        colors, count = np.unique(right_wind.reshape(-1,right_wind.shape[-1]),
                                  axis=0, return_counts=True)
        right = colors[count.argmax()]

        if np.sum(left) > np.sum(right):
            sides.append('right')
        else:
            sides.append('left')

        if len(sides) % 10 == 0:
            print('\t {:.4f}'.format(len(sides)/length))

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    f = open(folder_name + '/picture_sides/' +
             video_filename[:2]+'_picture_sides.txt', 'w+')

    f.write('Filename: ' + video_filename + '\n')
    for side in sides:
        f.write(side + ' \n')
    f.close()
    print('Wrote to data/picture_sides/' + video_filename[:2]+
          '_picture_sides.txt')

def generate_picture_side():
    folder_name = 'dataset2'
    video_filenames = os.listdir(folder_name + '/videos')
    for ii in range(len(video_filenames)):
        picture_side(folder_name, video_filenames[ii])

def filter_data(X, Y1, Y2, T, feat_names, feature_set, sensitivity_col):
    face_col_names = ['Head Proximity', 'Head Orientation', 'Gaze Direction', 'Eye Aspect Ratio', 'Pupil Ratio', 'AU04', 'AU07', 'AU12', 'AU25', 'AU26', 'AU45']
    context_col_names = ['Activity Ind', 'Video Time', 'Progress', 'Picture Side', 'Activity Type', 'Activity Time']
    success_col = feat_names.index('Success')
    confidence_col = feat_names.index('Confidence')

    cols_to_remove = [success_col, confidence_col]
    if not sensitivity_col == 'None':
        cols_to_remove.append(feat_names.index(sensitivity_col))
    if feature_set == 'context':
        for col_name in face_col_names:
            cols_to_remove.append(feat_names.index(col_name))
    if feature_set == 'face':
        for col_name in context_col_names:
            cols_to_remove.append(feat_names.index(col_name))

    cols_to_select = [x for x in range(len(feat_names)) if not x in cols_to_remove]

    new_X, new_Y1, new_Y2, new_T = [], [], [], []
    for xx, yy1, yy2, tt in zip(X, Y1, Y2, T):
        inds_to_keep = np.where(xx[:,success_col] > 0)[0]
        if inds_to_keep.shape[0] > 0:
            new_X.append(xx[inds_to_keep,:][:,cols_to_select])
            new_Y1.append(yy1[inds_to_keep])
            new_Y2.append(yy2[inds_to_keep])
            new_T.append(tt[inds_to_keep])
    return new_X, new_Y1, new_Y2, new_T, [feat_names[ii] for ii in cols_to_select]

def get_class_weight(Y, num_classes):
    class_weight = np.empty((num_classes))
    for ind in np.arange(num_classes):
        class_weight[ind] = np.sum(Y == ind)
    class_weight = class_weight/np.sum(class_weight)

    return class_weight

def find_model_split(T, num_models=4):
    lengths = [t[-1] for t in T]
    bins = np.interp(np.linspace(0, len(lengths), num_models + 1),np.arange(len(lengths)),np.sort(lengths))
    bins[-1] = -1
    bins[0] = 0
    return bins.astype('int')
