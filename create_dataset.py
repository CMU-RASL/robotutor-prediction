import os
import csv
import numpy as np
from pickle import dump

def read_feature_csv(filename, activity_name):
    time = []
    feat = []
    feedback = None
    backbutton = None
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for ii, row in enumerate(csv_reader):
            cur_feat = []
            if ii == 0:
                feedback = int(row[-3]) + 1
                backbutton = int(row[-1])
            elif ii == 1:
                headers = row
            else:
                #Time
                time.append(float(row[0]))

                #Ignore state (1)
                cur_feat = []
                for jj in range(2,len(row)):
                    cur_feat.append(float(row[jj]))
                feat.append(cur_feat)

    x = np.array(feat)
    t = np.array(time)
    y1 = (np.ones_like(t)*feedback).reshape(-1,1)
    y2 = (np.ones_like(t)*backbutton).reshape(-1,1)

    #Modifications
    if activity_name == 'story_hear':
        x = np.hstack((x, np.zeros_like(t).reshape(-1,1)))
    elif activity_name == 'story_read':
        x = np.hstack((x, np.ones_like(t).reshape(-1,1)))
    else: #echo
        x = np.hstack((x, np.ones_like(t).reshape(-1,1)))

    headers = headers[2:]
    headers.append('Activity')
    headers[headers.index('Lines')] = 'Progress'

    #Head orientation betwen -pi and pi
    x[np.where(x[:,1] > np.pi)[0],1] -= 2*np.pi

    return t, x, y1, y2, headers


def main():
    foldername = 'dataset2/dataset2'
    result_filename = 'dataset2.pkl'

    T = []
    X = []
    Y1 = []
    Y2 = []
    info = []

    for ii, filename in enumerate(os.listdir(foldername)):
        res = filename.split('_')
        vid_ind = res[0]
        activity_ind = res[1]
        activity_name = res[2] + '_' + res[3]
        activity_name = activity_name[:-4]

        if vid_ind:
            tt, xx, yy1, yy2, headers = read_feature_csv(foldername + '/' + filename, activity_name)
            T.append(tt)
            X.append(xx)
            Y1.append(yy1)
            Y2.append(yy2)
        if ii % 10 == 0:
            print('Finished {}/{}'.format(ii+1, len(os.listdir(foldername))))

    my_data = {'X': X, 'Y1': Y1, 'Y2': Y2, 'T': T, 'feat_names': headers}
    output = open(result_filename, 'wb')
    dump(my_data, output)
    output.close()
    print('Saved to:', result_filename)

if __name__ == '__main__':
    main()
