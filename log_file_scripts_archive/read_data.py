import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_feature_csv(filename):
    
    time = []
    feat = []
    feedback = None
    backbutton = None
    with open('data/allfeature/'+filename, mode='r') as csv_file:
      
        csv_reader = csv.reader(csv_file)
        
        for ii, row in enumerate(csv_reader):
            cur_feat = []
            if ii == 0:
                feedback = int(row[-3])
                backbutton = row[-1]
            elif ii == 1:
                headers = row
            else:
                for jj, entry in enumerate(row):
                    if jj == 0:
                        time.append(float(entry))
                    elif jj == 1:
                        if entry == 'listen':
                            cur_feat.append(0.0)
                        else:
                            cur_feat.append(1.0)
                    elif jj == 8:
                        if entry == 'left':
                            cur_feat.append(0.0)
                        else:
                            cur_feat.append(1.0)
                    else:
                        cur_feat.append(float(entry))
                feat.append(cur_feat)
            
#            if ii % 1000 == 0:
#                print('Processed', ii)
                
    time = np.array(time)
    feat = np.array(feat)
    
    return time, feat, feedback, backbutton, headers[1:]

def remove_outliers(x):
    upper_quartile = np.percentile(x, 75)
    lower_quartile = np.percentile(x, 25)
    IQR = (upper_quartile - lower_quartile) * 2.0
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    
    result1 = np.where(x <= quartileSet[0])[0]
    result2 = np.where(x >= quartileSet[1])[0]
    
    x[result1] = 0.0
    x[result2] = 0.0
    
    return x

def get_start_end_inds(start_val, end_val, t):
    
    if not start_val == 0:
        start_ind = np.argmin(abs(t[-1] - t - start_val))
    else:
        start_ind = 0
        
    if not end_val == -1:
        end_ind = np.argmin(abs(t - t[0] - end_val))
    else:
        end_ind = t.shape[0]
    
    if start_ind > end_ind:
        start_ind, end_ind = end_ind, start_ind
        
    return start_ind, end_ind

def get_model_data(T,X,Y1,Y2,info,headers,start_val=0,end_val=-1):
    
    T_arr = []
    X_arr = []
    Y1_arr = []
    Y2_arr = []
    
    for ii in range(len(X)):
        if info[ii][2][:8] == 'writewrd':
            activity_type = 0
        elif info[ii][2][:9] == 'storyhear':
            activity_type = 1
        else:
            activity_type = 2
        
        start_ind, end_ind = get_start_end_inds(start_val, end_val, T[ii])
        
        cur_X = X[ii][start_ind:end_ind,:]
        cur_Y1 = Y1[ii]*np.ones((cur_X.shape[0],1)) + 1.0
        
        if Y2[ii] == 'True':
            cur_Y2 = np.ones((cur_X.shape[0],1))
        else:
            cur_Y2 = np.zeros((cur_X.shape[0],1))
            
            
        cur_T = T[ii][start_ind:end_ind]
        cur_T = cur_T - cur_T[0]
        cur_X = np.hstack((cur_X, np.ones_like(cur_Y1)*activity_type))
        
        ind = headers.index('Lines')
        cur_X[:,ind] = cur_X[:,ind] - cur_X[0,ind]
    
        ind = headers.index('Picture Side')
        if activity_type == 0:
            cur_X[:,ind] = 0
        
        ind = headers.index('Eye Aspect Ratio')
        cur_X[:,ind] = np.maximum(cur_X[:,ind], -0.1)
        cur_X[:,ind] = np.minimum(cur_X[:,ind], 0.1)
        
        ind = headers.index('Pupil Ratio')
        cur_X[:,ind] = np.maximum(cur_X[:,ind], -0.3)
        cur_X[:,ind] = np.minimum(cur_X[:,ind], 0.3)
        
        ind = headers.index('Head Orientation')
        change_inds = np.where(cur_X[:,ind] > np.pi)
        cur_X[change_inds,ind] -= 2*np.pi
        
        T_arr.append(cur_T)
        X_arr.append(cur_X)
        Y1_arr.append(cur_Y1)
        Y2_arr.append(cur_Y2)
        
    return T_arr, X_arr, Y1_arr, Y2_arr

def plot_cdf(z, title):
    
    fig, ax = plt.subplots(2,1,figsize=(8, 8))
    x = np.sort(z)
    y = np.array(range(len(z)))/float(len(z))
    
    ax[0].plot(x, y, 'o-b')
    
    perc = [np.percentile(x, ii) for ii in [20, 40, 60, 80]]
    
    text = ' Percentiles:\n 0%     {:.1f}\n 20%   {:.1f}\n 40%   {:.1f}\n 60%   {:.1f}\n 80%   {:.1f}\n 100% {:.1f}'.format(0, perc[0], perc[1], perc[2], perc[3], x[-1])
    
    ax[0].set_xlim(0,800)
    ax[0].set_title(title, fontsize=20)
    ax[0].set_ylabel('CDF', fontsize=14)
    ax[0].text(0.75, 0.1, text, color="k", fontsize=14,
                 transform=ax[0].transAxes)
    
    ax[1].boxplot(z, vert=False, showfliers=True, widths = 0.65)
    ax[1].set_ylabel('Boxplot', fontsize=14)
    ax[1].set_xlim(0,800)
    ax[1].set_yticklabels([])
    ax[1].set_xlabel('Time (s)', fontsize=14)
    plt.show()

def create_cdfs(T, X, Y1, Y2):
    
    bail_len = []
    completed_len = []
    all_len = []
    
    for ii in range(len(X)):
        
        if Y2[ii][0] == 0:
            completed_len.append(T[ii][-1])
        else:
            bail_len.append(T[ii][-1])
        all_len.append(T[ii][-1])
    
    plot_cdf(bail_len, 'Bailed')
    plot_cdf(completed_len, 'Completed')
    plot_cdf(all_len, 'All')
    

def main():
    
    T = []
    X = []
    Y1 = []
    Y2 = []
    info = []
    
    for filename in os.listdir('data/allfeature'):
        res = filename.split('_')
        vid_ind = res[0]
        activity_ind = res[1]
        activity_name = res[2] + '.' + res[3]
    
        if vid_ind:
            time, feat, feedback, backbutton, headers = read_feature_csv(filename)
            if time.shape[0] > 1:          
                T.append(time)
                X.append(feat)
                Y1.append(feedback)
                Y2.append(backbutton)
                info.append([vid_ind, activity_ind, activity_name])
    
    model_T, model_X, model_Y1, model_Y2 = get_model_data(T,X,Y1,Y2,info,headers)
    
    create_cdfs(model_T, model_X, model_Y1, model_Y2)
    #np.savez('all_data.npz', X=model_X, Y1=model_Y1, Y2=model_Y2, T=model_T)

    
if __name__ == "__main__":
    main()