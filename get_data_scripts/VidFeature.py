from datetime import datetime, timedelta
import numpy as np
import numpy.linalg
import scipy.spatial.transform

class VidFeature():
    
    def __init__(self, filename, start):
        self.start = start
        self.head_prox = []
        self.head_orient = []
        self.gaze_dir = []
        self.eye_aspect_ratio = []
        self.pupil_ratio = []
        self.timestamps = []
        self.sides = []
        self.filename = filename
    
    def print_features(self):
        
        f = open('data/vidfeature/' + self.filename[:2]+'_vidfeature.txt', 'w+')
        
        f.write('Filename: ' + self.filename + '\n')
        for ii in range(len(self.timestamps)):
            f.write(self.timestamps[ii].strftime("%H:%M:%S.%f") + \
                    ', Head proximity (m): ' + str(np.round(self.head_prox[ii], decimals=2)) + \
                    ', Head orient magnitude (deg): ' + str(np.round(self.head_orient[ii][1]*180/np.pi, decimals=2)) + \
                    ', Gaze dir magnitude (deg): ' + str(np.round(self.gaze_dir[ii][1]*180/np.pi, decimals=2)) + \
                    ', Eye aspect ratio: ' + str(np.round(self.eye_aspect_ratio[ii], decimals=2)) + \
                    ', Pupil ratio: ' + str(np.round(self.pupil_ratio[ii], decimals=2)) + \
                    ', Picture side: ' + self.sides[ii] + \
                    ' \n')
        
        
        f.close()
        print('Wrote to file:', 'data/vidfeature/' + self.filename[:2]+'_vidfeature.txt')   
    
    def add_line(self, headers, line):
        
        #Timestamp
        ind = headers.index('timestamp')
        sec = line[ind].rstrip().strip()
        self.timestamps.append(self.start + timedelta(seconds=float(sec)))
        
        #Head Prox
        head_prox_feat = ['pose_Tx', 'pose_Ty', 'pose_Tz']
        head_prox = np.empty((3))

        for ii, cur_feat in enumerate(head_prox_feat):
            ind = headers.index(cur_feat)
            if ind > len(line)-1:
                head_prox[ii] = 0.0
            else:
                head_prox[ii] = float(line[ind].strip())
        self.head_prox.append(np.linalg.norm(head_prox))
        
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
        self.head_orient.append((rot_vec, rot_vec_mag))
        
        #Gaze dir
        gaze_dir_feat = ['gaze_angle_x', 'gaze_angle_y']
        gaze_vec = np.empty((2))
        for ii, cur_feat in enumerate(gaze_dir_feat):
            ind = headers.index(cur_feat)
            if ind > len(line)-1:
                gaze_vec[ii] = 0.0
            else:
                gaze_vec[ii] = float(line[ind].strip())
        gaze_mag = np.linalg.norm(gaze_vec)
        self.gaze_dir.append((gaze_vec, gaze_mag))
        
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
        
        self.eye_aspect_ratio.append(np.mean((Er_left, Er_right)))

        Pr_left = np.linalg.norm(l_l - j_l) * np.linalg.norm(k_l - i_l) \
            / (np.linalg.norm(e_l - a_l) * np.linalg.norm(g_l - c_l) + 1e-5)
        Pr_right = np.linalg.norm(l_r - j_r) * np.linalg.norm(k_r - i_r) \
            / (np.linalg.norm(e_r - a_r) * np.linalg.norm(g_r - c_r) + 1e-5)

        self.pupil_ratio.append(np.mean((Pr_left, Pr_right)))
        
    def get_activity_features(self, start, stop):
        
        video_ind_start = len(self.timestamps)
        video_ind_end = len(self.timestamps)
        
        for ii, timestamp in enumerate(self.timestamps):
            if timestamp > start:
                video_ind_start = min(video_ind_start, ii)
            if timestamp > stop:
                video_ind_end = min(video_ind_end, ii)
        
        cur_timestamps = self.timestamps[video_ind_start:video_ind_end]
        
        num_points = len(cur_timestamps)
        
        activity_features = {'timestamps': cur_timestamps, \
                             'head_prox': np.empty(num_points), \
                             'head_orient': np.empty(num_points), \
                             'gaze_dir': np.empty(num_points), \
                             'eye_aspect_ratio': np.empty(num_points), \
                             'pupil_ratio': np.empty(num_points), \
                             'sides': np.empty(num_points)}
        
        for ii, ind in enumerate(range(video_ind_start, video_ind_end)):
            activity_features['head_prox'][ii] = self.head_prox[ind]/1000.0
            activity_features['head_orient'][ii] = self.head_orient[ind][1]
            activity_features['gaze_dir'][ii] = self.gaze_dir[ind][1]
            activity_features['eye_aspect_ratio'][ii] = self.eye_aspect_ratio[ind]
            activity_features['pupil_ratio'][ii] = self.pupil_ratio[ind]
            if self.sides[ii] == 'left':
                activity_features['sides'][ii] = 0
            else:
                activity_features['sides'][ii] = 1
                
        return activity_features
        
    def add_sides(self, sides):
        
        if len(sides) < len(self.timestamps):
            for ii in range(len(self.timestamps) - len(sides)):
                sides.append('')
        self.sides = sides