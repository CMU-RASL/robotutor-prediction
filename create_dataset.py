import os
import csv
import numpy as np
from pickle import dump

def read_feature_csv(filename, activity_name, activity_num):
    time = []
    feat = []
    feedback = None
    backbutton = None

    with open(filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for ii, row in enumerate(csv_reader):
            cur_feat = []
            if ii == 0:
                feedback = int(row[-3])
                backbutton = int(row[-1])
            elif ii == 2:
                headers = row[1:]
            elif ii > 3 and len(row) > 1:
                #Time
                time.append(float(row[0]))

                cur_feat = []
                for jj in range(1,len(row)):
                    cur_feat.append(float(row[jj]))
                feat.append(cur_feat)
    x = np.array(feat)
    t = np.array(time)
    x = np.hstack((x, t.reshape(-1,1)))
    y1 = (np.ones_like(t)*feedback).reshape(-1,1).astype('int')
    y2 = (np.ones_like(t)*backbutton).reshape(-1,1).astype('int')
    headers.append('Activity Time')
    return t, x, y1, y2, headers

def get_num_activities(foldername):
    num_activities = {}
    for ii, filename in enumerate(os.listdir(foldername)):
        res = filename.split('_')
        vid_ind = int(res[0])

        if vid_ind in num_activities:
            num_activities[vid_ind] += 1
        else:
            num_activities[vid_ind] = 1
    return num_activities

def main():
    foldername = 'dataset2/csvs'
    result_filename = 'dataset2_mod.pkl'

    T = []
    X = []
    Y1 = []
    Y2 = []
    # info = []
    tots = np.zeros((2, 3))
    num_activities = get_num_activities(foldername)
    for ii, filename in enumerate(os.listdir(foldername)):
        res = filename.split('_')
        vid_ind = res[0]
        activity_ind = res[1]
        activity_name = res[2] + '_' + res[3]
        activity_name = activity_name[:-4]

        if vid_ind and float(activity_ind) < num_activities[int(vid_ind)]:
            tt, xx, yy1, yy2, headers = read_feature_csv(foldername + '/' + filename, activity_name, float(activity_ind)/num_activities[int(vid_ind)])
            if np.sum(tots[0,:]) < 20 or (tots[0,yy1[0,0]]/np.sum(tots[0,:]) < 0.6 and tots[1,yy2[0,0]]/np.sum(tots[0,:]) < 0.6):
                T.append(tt)
                X.append(xx)
                Y1.append(yy1)
                Y2.append(yy2)

                tots[0, yy1[0,0]] += 1
                tots[1, yy2[0,0]] +=1
                print(np.round(tots[0,:]/np.sum(tots[0,:])*100, decimals=2), np.round(tots[1,:]/np.sum(tots[0,:])*100, decimals=2))
        # if ii % 10 == 0:
        #     print('Finished {}/{}'.format(ii+1, len(os.listdir(foldername))))

    print('Total Number of Activities', len(X))
    my_data = {'X': X, 'Y1': Y1, 'Y2': Y2, 'T': T, 'feat_names': headers}
    output = open(result_filename, 'wb')
    dump(my_data, output)
    output.close()
    print('Saved to:', result_filename)

if __name__ == '__main__':
    main()
