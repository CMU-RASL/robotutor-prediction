import numpy as np
import os
import cv2

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

def get_prob(model, x, num_classes):
    prob = np.zeros(num_classes).astype('float')
    pred = model.predict_proba(x).flatten()
    for ii, class_ind in enumerate(model.classes_):
        prob[int(class_ind)] = pred[ii]
    return prob

def get_metrics(conf_mat, num_classes):
    if num_classes == 2:
        cur_class = 1
        not_cur_class = 0
        tp = conf_mat[cur_class,cur_class].astype('float')
        tn = conf_mat[not_cur_class,not_cur_class].astype('float')
        fp = conf_mat[not_cur_class,cur_class].astype('float')
        fn = conf_mat[cur_class,not_cur_class].astype('float')
        tpr = tp/(tp + fn + 1e-6)
        fpr = fp/(fp + tn + 1e-6)
        acc = (tp + tn)/(tp + tn + fp + fn + 1e-6)

    else:
        tpr = np.empty(3)
        fpr = np.empty(3)
        acc = np.empty(3)
        for cur_class in range(num_classes):
            not_cur_class = list(range(num_classes))
            not_cur_class.remove(cur_class)
            tp = conf_mat[cur_class,cur_class].astype('float')
            tn = np.sum(conf_mat[not_cur_class,not_cur_class]).astype('float')
            fp = np.sum(conf_mat[not_cur_class,cur_class]).astype('float')
            fn = np.sum(conf_mat[cur_class,not_cur_class]).astype('float')
            tpr[cur_class] = tp/(tp + fn + 1e-6)
            fpr[cur_class] = fp/(fp + tn + 1e-6)
            acc[cur_class] = (tp + tn)/(tp + tn + fp + fn + 1e-6)

    return fpr, tpr, acc

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
