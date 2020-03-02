import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy.linalg
import scipy.spatial.transform
from skimage.measure import compare_ssim
import scipy.misc


def get_picture_side(frame):
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
#        print('Pic on right')
        side = 1
    else:
#        print('Pic on left')
        side = 0
    return side

def get_read_fraction(frame, picture_side):
    
    height, width, channels = frame.shape

    if picture_side == 1: #picture on left
        wind = frame[160: height-100, 
                       0: width//2 - 25,:]
    else: #right
        wind = frame[160: height-100, 
                       width // 2 + 25: width,:]
        
    res = np.where((wind[:,:,0] > 0) & (wind[:,:,1] > 100) & (wind[:,:,2] > 0)
                    & (wind[:,:,0] < 150) & (wind[:,:,1] < 256) & (wind[:,:,2] < 150))
    total_read = res[0].shape[0]
  
    res2 = np.where((wind[:,:,0] < 200) & (wind[:,:,1] < 200) & (wind[:,:,2] < 200))
    total_text = res2[0].shape[0]

    frac = np.clip(total_read / total_text, 0, 1)
    return frac

def get_openface_features(headers, line):

    #Timestamp
    ind = headers.index('timestamp')
    sec = float(line[ind].rstrip().strip())
    
    #Head Prox
    head_prox_feat = ['pose_Tx', 'pose_Ty', 'pose_Tz']
    head_prox = np.empty((3))

    for ii, cur_feat in enumerate(head_prox_feat):
        ind = headers.index(cur_feat)
        if ind > len(line)-1:
            head_prox[ii] = 0.0
        else:
            head_prox[ii] = float(line[ind].strip())
    head_prox = np.linalg.norm(head_prox/1000)
    if head_prox > 1000:
        head_prox = head_prox/10e4
    
    
    #Head Orient
    head_orient_feat = ['pose_Rx', 'pose_Ry', 'pose_Rz']
    head_orient = np.empty((3))
    for ii, cur_feat in enumerate(head_orient_feat):
        ind = headers.index(cur_feat)
        if ind > len(line)-1:
            head_orient[ii] = 0.0
        else:
            head_orient[ii] = float(line[ind].strip())
    rot = scipy.spatial.transform.Rotation.from_euler('xyz', head_orient)
    rot_vec = rot.as_rotvec()
    rot_vec_mag = np.linalg.norm(rot_vec)
    head_orient = rot_vec_mag
    
    #Gaze dir
    gaze_dir_feat = ['gaze_angle_x', 'gaze_angle_y']
    gaze_vec = np.empty((2))
    for ii, cur_feat in enumerate(gaze_dir_feat):
        ind = headers.index(cur_feat)
        if ind > len(line)-1:
            gaze_vec[ii] = 0.0
        else:
            gaze_vec[ii] = float(line[ind].strip())
    gaze_dir = np.linalg.norm(gaze_vec)
    
    #[a, b, c, d, e, f, g, h, i, j, k, l]
    left_feat = [8, 10, 11, 13, 14, 16, 17, 19, 25, 23, 21, 27]
    right_feat = [36, 38, 39, 41, 42, 44, 45, 47, 53, 51, 49, 55]
    
    left_eye = np.empty((len(left_feat), 2))
    right_eye = np.empty((len(left_feat), 2))
    
    for ii in range(len(left_feat)):
        ind = headers.index('eye_lmk_x_' + str(left_feat[ii]))
        if ind > len(line)-1:
            left_eye[ii, 0] = 0.0
            left_eye[ii,1] = 0.0
            right_eye[ii,0] = 0.0
            right_eye[ii,1] = 0.0
        else:
            left_eye[ii,0] = float(line[ind].strip())
            ind = headers.index('eye_lmk_y_' + str(left_feat[ii]))
            left_eye[ii,1] = float(line[ind].strip())
            ind = headers.index('eye_lmk_x_' + str(right_feat[ii]))
            right_eye[ii,0] = float(line[ind].strip())
            ind = headers.index('eye_lmk_y_' + str(right_feat[ii]))
            right_eye[ii,1] = float(line[ind].strip())
    
    a_l, a_r = left_eye[0,:], right_eye[0,:]
    b_l, b_r = left_eye[1,:], right_eye[1,:]
    c_l, c_r = left_eye[2,:], right_eye[2,:]
    d_l, d_r = left_eye[3,:], right_eye[3,:]
    e_l, e_r = left_eye[4,:], right_eye[4,:]
    f_l, f_r = left_eye[5,:], right_eye[5,:]
    g_l, g_r = left_eye[6,:], right_eye[6,:]
    h_l, h_r = left_eye[7,:], right_eye[7,:]
    i_l, i_r = left_eye[8,:], right_eye[8,:]
    j_l, j_r = left_eye[9,:], right_eye[9,:]
    k_l, k_r = left_eye[10,:], right_eye[10,:]
    l_l, l_r = left_eye[11,:], right_eye[11,:]
    
    Er_left = (np.linalg.norm(h_l - b_l) - np.linalg.norm(f_l - d_l)) \
        / (2 * np.linalg.norm(e_l - a_l) + 1e-5)
    Er_right = (np.linalg.norm(h_r - b_r) - np.linalg.norm(f_r - d_r)) \
        / (2 * np.linalg.norm(e_r - a_r) + 1e-5)
    
    eye_aspect_ratio = np.mean((Er_left, Er_right))

    Pr_left = np.linalg.norm(l_l - j_l) * np.linalg.norm(k_l - i_l) \
        / (np.linalg.norm(e_l - a_l) * np.linalg.norm(g_l - c_l) + 1e-5)
    Pr_right = np.linalg.norm(l_r - j_r) * np.linalg.norm(k_r - i_r) \
        / (np.linalg.norm(e_r - a_r) * np.linalg.norm(g_r - c_r) + 1e-5)

    pupil_ratio = np.mean((Pr_left, Pr_right))
    
    return sec, head_prox, head_orient, gaze_dir, eye_aspect_ratio, pupil_ratio


def get_activity_type(frame):
    
    height, width, channels = frame.shape
    wind = frame[475: 675, 0: 500, :]
    
    crop_image = wind[20:40,200:270,:]
#    scipy.misc.imsave('templates/story_read.png',crop_image)
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    
    activity_names = []
    ssims = []
    
    for filename in os.listdir('templates/activities'):
        template_image = cv2.imread('templates/activities/'+filename,cv2.IMREAD_GRAYSCALE)
        activity_names.append(filename[:-4])
        ssims.append(abs(compare_ssim(template_image, gray)))
    
    ssims = np.array(ssims)
    if np.max(ssims) < 0.7:
        return 'n/a'
    else:
        activity_name = activity_names[np.argmax(ssims)]
        return activity_name

def get_feedback(frame):
    
    positive_image = frame[100:200,100:200,:]
    neutral_image = frame[400:500,100:200,:]
    negative_image = frame[600:700,100:200,:]

#    scipy.misc.imsave('templates/feedbacks/neutral.png',neutral_image)
    ssims = np.zeros(3)
    
    if 'negative.png' in os.listdir('templates/feedbacks'):
        gray = cv2.cvtColor(negative_image, cv2.COLOR_BGR2GRAY)
        template_image = cv2.imread('templates/feedbacks/negative.png',cv2.IMREAD_GRAYSCALE)
        ssims[0] = compare_ssim(template_image, gray)
    if 'neutral.png' in os.listdir('templates/feedbacks'):
        gray = cv2.cvtColor(neutral_image, cv2.COLOR_BGR2GRAY)
        template_image = cv2.imread('templates/feedbacks/neutral.png',cv2.IMREAD_GRAYSCALE)
        ssims[1] = compare_ssim(template_image, gray)
    if 'positive.png' in os.listdir('templates/feedbacks'):
        gray = cv2.cvtColor(positive_image, cv2.COLOR_BGR2GRAY)
        template_image = cv2.imread('templates/feedbacks/positive.png',cv2.IMREAD_GRAYSCALE)
        ssims[2] = compare_ssim(template_image, gray)
    
    if np.max(ssims) < 0.9:
        return -1
    else:
        return np.argmax(np.abs(ssims))

def process_video(video_filename, csv_filename):
    cap = cv2.VideoCapture('data2/videos/'+video_filename)
    cap.get(7)
    cap.set(1, 100)
    res, frame = cap.read()
    
#    plt.imshow(frame)
    
    #activity selector screen 80(hear), 2850 (echo), 17000 (hear)
    activity_name = get_activity_type(frame)
    print('Activity Name', activity_name)
#    
#    #within activity  400-1000
#    picture_side = get_picture_side(frame)
#    read_fraction = get_read_fraction(frame, picture_side)
#    print('Picture side', picture_side)
#    print('Read Fraction:', read_fraction)

    #feedback screen 2600 (1), 15850 (1), 15800 (-1, nothing selected)
    feedback = get_feedback(frame)
    print('Feedback', feedback)
    
#    with open('data2/openface/' + csv_filename, mode='r') as csv_file:
#        csv_reader = csv.reader(csv_file, delimiter=',')
#        for ii, line in enumerate(csv_reader):
#            if ii == 0:
#                headers = [s.strip() for s in line]
#            else:
#                openface_features = get_openface_features(headers, line)

    cap.release()
    cv2.destroyAllWindows()


def main():
    video_filename = '0001.mp4'
    csv_filename = '0001.csv'
    
    process_video(video_filename, csv_filename)

if __name__ == '__main__':
    main()
