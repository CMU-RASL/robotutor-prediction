import os
import csv
import numpy as np

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
                backbutton = row[-1]
                if backbutton[0] == 'U':
                    backbutton = None
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
    y2 = (np.zeros_like(t)).reshape(-1,1)

    #Modifications
    if activity_name == 'story_hear':
        x = np.hstack((x, np.zeros_like(t).reshape(-1,1)))
    else: #echo
        x = np.hstack((x, np.ones_like(t).reshape(-1,1)))

    headers = headers[2:]
    headers.append('Activity')
    headers[5] = 'Progress'

    #Eye aspect ratio between -0.1 and 0.1
    x[:,3] = np.clip(x[:,3], -0.1, 0.1)

    #Pupil aspect ratio between -0.3 and 0.3
    x[:,4] = np.clip(x[:,4], -0.3, 0.3)

    #Head orientation betwen -pi and pi
    x[np.where(x[:,1] > np.pi)[0],1] -= 2*np.pi

    return t, x, y1, y2, headers


def main():
    foldername = 'dataset1/dataset1_vid'
    npz_name = 'dataset1_vid.npz'
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
        print('Finished {}/{}'.format(ii+1, len(os.listdir(foldername))))
    print(headers)
    np.savez(npz_name, X=X, Y1=Y1, Y2=Y2, T=T)
    print('Saved to {}'.format(npz_name))

if __name__ == '__main__':
    main()
