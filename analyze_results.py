import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper import class_name_from_ind
from model_training import load_model
import os
from mpl_toolkits.axes_grid1 import ImageGrid

def hypothesisI():
    for alpha, beta in zip([0.6, 1.0], [0.2, 0.0]):
        with open('result//result_alpha_' + str(alpha) + '_beta_' + str(beta) + '_features_all.pkl', 'rb') as f:
            all_feat = pickle.load(f)
        with open('result//result_alpha_' + str(alpha) + '_beta_' + str(beta) + '_features_face.pkl', 'rb') as f:
            face_feat = pickle.load(f)
        with open('result//result_alpha_' + str(alpha) + '_beta_' + str(beta) + '_features_context.pkl', 'rb') as f:
            context_feat = pickle.load(f)

        #Find best indices
        k = 10
        cv0 = 1.383
        cv1 = 1.833 #one-sided t-test with 9 df
        cv2 = 2.262 #two-sided t-test with 9 df
        cv1_1 = 2.821 #one-sided t-test with 9 df
        cv2_1 = 3.250 #two-sided t-test with 9 df
        legend = ['Feedback', 'Backbutton']

        fig, axs = plt.subplots(1, 2, figsize=(9,4))
        for modeltype_ind in range(2):
            print(legend[modeltype_ind])

            metric_all_feat = all_feat['best_vals'][modeltype_ind][0]

            metric_face_feat = face_feat['best_vals'][modeltype_ind][0]

            metric_context_feat = context_feat['best_vals'][modeltype_ind][0]
            data_to_plot = [metric_face_feat, metric_context_feat, metric_all_feat]
            # print([(np.min(feat), np.max(feat)) for feat in data_to_plot])

            p = metric_all_feat - metric_face_feat
            pbar = np.mean(p)
            den = np.sum((p-pbar)**2)/(k-1)
            tstat = pbar*np.sqrt(k) / np.sqrt(den)
            # interpret via critical value
            if abs(tstat) <= cv1:
                if abs(tstat) <= cv0:
                   print('\tFace and Context + Face have the same performance')
                else:
                   print('\tFace and Context + Face 0.1 level')
            else:
                if abs(tstat) <= cv1_1:
                    print('\tFace and Context + Face 0.05 level')
                else:
                    print('\tFace and Context + Face 0.01 level')

            p = metric_all_feat - metric_context_feat
            pbar = np.mean(p)
            den = np.sum((p-pbar)**2)/(k-1)
            tstat = pbar*np.sqrt(k) / np.sqrt(den)
            # interpret via critical value
            if abs(tstat) <= cv1:
                if abs(tstat) <= cv0:
                   print('\tContext and Context + Face have the same performance')
                else:
                   print('\tContext and Context + Face 0.1 level')
            else:
                if abs(tstat) <= cv1_1:
                    print('\tContext and Context + Face 0.05 level')
                else:
                    print('\tContext and Context + Face 0.01 level')

            p = metric_context_feat - metric_face_feat
            pbar = np.mean(p)
            den = np.sum((p-pbar)**2)/(k-1)
            tstat = pbar*np.sqrt(k) / np.sqrt(den)
            # interpret via critical value
            if abs(tstat) <= cv2:
                if abs(tstat) <= cv0:
                   print('\tContext and Face have the same performance')
                else:
                   print('\tContext and Face 0.1 level')
            else:
                if abs(tstat) <= cv2_1:
                    print('\tContext and Face 0.05 level')
                else:
                    print('\tContext and Face 0.01 level')

            axs[modeltype_ind].boxplot(data_to_plot)
            axs[modeltype_ind].set_title(legend[modeltype_ind])
            axs[modeltype_ind].set_xticklabels(['Face\nOnly', 'Context\nOnly', 'Face and\nContext'])
            axs[modeltype_ind].set_ylim([0.6, 1])
            axs[modeltype_ind].set_ylabel('S')
        break
    plt.show()

def hypothesisII():
    for alpha, beta in zip([0.6, 1.0], [0.2, 0.0]):
        with open('result//result_alpha_' + str(alpha) + '_beta_' + str(beta) + '_features_all_modelnum1.pkl', 'rb') as f:
            model1 = pickle.load(f)
        with open('result//result_alpha_' + str(alpha) + '_beta_' + str(beta) + '_features_all_modelnum26.pkl', 'rb') as f:
            model26 = pickle.load(f)

        #Find best indices
        k = 10

        cv0 = 1.383
        cv1 = 1.833 #one-sided t-test with 9 df
        cv1_1 = 2.821 #one-sided t-test with 9 df
        cv2 = 2.262 #two-sided t-test with 9 df
        legend = ['Feedback', 'Backbutton']

        fig, axs = plt.subplots(1, 2, figsize=(6,4))
        for modeltype_ind in range(2):
            print(legend[modeltype_ind])
            metric_1 = model1['best_vals'][modeltype_ind][0]
            metric_2 = model26['best_vals'][modeltype_ind][0]

            p = metric_2 - metric_1
            pbar = np.mean(p)

            den = np.sum((p-pbar)**2)/(k-1)
            if den < 1e-6:
                tstat = 0.0
            else:
                tstat = pbar*np.sqrt(k) / np.sqrt(den)

            # interpret via critical value
            if abs(tstat) <= cv1:
                print('\tOne interval and more than one interval have the same performance')
            else:
                if abs(tstat) <= cv1_1:
                    print('\tOne interval and more than one interval 0.05 level')
                else:
                    print('\tOne interval and more than one interval 0.01 level')
            print('')
            data_to_plot = [metric_1, metric_2]
            axs[modeltype_ind].boxplot(data_to_plot)
            axs[modeltype_ind].set_title(legend[modeltype_ind])
            axs[modeltype_ind].set_xticklabels(['1 Time\nInterval', 'More than 1\nTime Interval'])
            axs[modeltype_ind].set_ylim([0.5, 1])
            axs[modeltype_ind].set_ylabel('S')
    plt.show()

def hypothesisIII():
    for alpha, beta in zip([1.0], [0.0]):
        accs = []
        with open('result//result_alpha_' + str(alpha) + '_beta_' + str(beta) + '_features_all_casenum_1.pkl', 'rb') as f:
            result = pickle.load(f)
            accs.append(result['best_vals'])
        with open('result//result_alpha_' + str(alpha) + '_beta_' + str(beta) + '_features_all.pkl', 'rb') as f:
            result = pickle.load(f)
            accs.append(result['best_vals'])

        #Find best indices
        k = 10

        cv0 = 1.383
        cv1 = 1.833
        cv2 = 2.821
        legend = ['Feedback', 'Backbutton']
        fig, axs = plt.subplots(1, 2, figsize=(6,4))
        for modeltype_ind in range(2):

            print(legend[modeltype_ind])
            metric_1 = accs[0][modeltype_ind][0]
            metric_2 = accs[1][modeltype_ind][0]
            print(np.mean(metric_1), np.mean(metric_2))
            #Check all significance
            p = metric_2 - metric_1
            pbar = np.mean(p)
            den = np.sum((p-pbar)**2)/(k-1)
            if den < 1e-6:
                tstat = 0.0
            else:
                tstat = pbar*np.sqrt(k) / np.sqrt(den)
            # interpret via critical value
            if abs(tstat) <= cv0:
                print('\tGuessing and algorithm have same performance')
            else:
                if abs(tstat) <= cv1:
                    print('\tGuessing and algorithm - 0.1 level')
                else:
                    if abs(tstat) <= cv2:
                        print('\tGuessing and algorithm - 0.05 level')
                    else:
                        print('\tGuessing and algorithm - 0.01 level')

            axs[modeltype_ind].boxplot([metric_1, metric_2])
            axs[modeltype_ind].set_title(legend[modeltype_ind])
            axs[modeltype_ind].set_xticklabels(['Guessing','Algorithm'])
            axs[modeltype_ind].set_ylim([0.0, 1.0])
            axs[modeltype_ind].set_ylabel('S')
    # plt.show()

def weights():
    acc_grid = np.zeros((2, 11, 11))
    early_grid = np.zeros((2, 11, 11))
    freq_grid = np.zeros((2, 11, 11))
    alpha_arr = np.arange(0, 1.1, 0.1)
    beta_arr = np.arange(0, 1.1, 0.1)
    for alpha_ind, alpha in enumerate(alpha_arr):
        for beta_ind, beta in enumerate(beta_arr):
            if 1 - alpha - beta >= -1e-4:
                result_filename = 'result//result_alpha_{}_beta_{}_features_{}.pkl'.format(alpha, beta, 'all')
                with open(result_filename, 'rb') as f:
                    res = pickle.load(f)
                    acc_grid[0, alpha_ind, beta_ind], acc_grid[1, alpha_ind, beta_ind] = np.mean(res['best_vals'][0][1]), np.mean(res['best_vals'][1][1])
                    early_grid[0, alpha_ind, beta_ind], early_grid[1, alpha_ind, beta_ind] = np.mean(res['best_vals'][0][2]), np.mean(res['best_vals'][1][2])
                    freq_grid[0, alpha_ind, beta_ind], freq_grid[1, alpha_ind, beta_ind] = np.mean(res['best_vals'][0][3]), np.mean(res['best_vals'][1][3])
    grids = [acc_grid, early_grid, freq_grid]
    titles = ['Accuracy', 'Earliness', 'Frequency']
    fig, ax = plt.subplots(2, 3)
    for row in range(2):
        for col in range(3):
            im = ax[row, col].imshow(grids[col][row,:,:], vmin=0.4)
            # We want to show all ticks...
            ax[row, col].set_xticks(np.arange(len(alpha_arr)))
            ax[row, col].set_yticks(np.arange(len(beta_arr)))
            # ... and label them with the respective list entries
            ax[row, col].set_xticklabels(np.round(alpha_arr, decimals=1))
            ax[row, col].set_yticklabels(np.round(beta_arr, decimals=1))
            ax[row, col].set_xlabel('Alpha')
            ax[row, col].set_ylabel('Beta')

            # Loop over data dimensions and create text annotations.
            for i in range(len(alpha_arr)):
                for j in range(len(beta_arr)):
                    if grids[col][row,i,j] > 0.0:
                        text = ax[row,col].text(j, i, np.round(grids[col][row,i,j], decimals=2),
                                       ha="center", va="center", color="k")
            ax[0, col].set_title(titles[col])
    fig.tight_layout()
    plt.show()

def sensitivity():
    cols = ['Activity Ind', 'Video Time', 'Head Proximity', 'Head Orientation',
            'Gaze Direction', 'Eye Aspect Ratio', 'Pupil Ratio', 'AU04', 'AU07', 'AU12',
            'AU25', 'AU26', 'AU45', 'Progress', 'Picture Side', 'Activity Type', 'Activity Time']
    col_names = ['None', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
    alpha = 0.6
    beta = 0.2
    best_params = [[1, 6, 0.65, 0.0], [6, 6, 0.55, -0.1]]
    with open('result//result_alpha_' + str(alpha) + '_beta_' + str(beta) + '_features_all.pkl', 'rb') as f:
        all_feat = pickle.load(f)

    results = []
    for col in cols:
        with open('result//result_alpha_' + str(alpha) + '_beta_' + str(beta) + '_features_all_remove_'+col+'.pkl', 'rb') as f:
            results.append(pickle.load(f))

    #Find best indices
    k = 10

    cv1 = 1.833 #one-sided t-test with 9 df
    cv2 = 2.262 #two-sided t-test with 9 df
    cv1_1 = 2.821 #one-sided t-test with 9 df
    cv2_1 = 3.250 #two-sided t-test with 9 df
    legend = ['Feedback', 'Backbutton']
    fig, axs = plt.subplots(1, 2, figsize=(10,6))

    for modeltype_ind in range(2):
        print(legend[modeltype_ind])
        ind1 = np.where(all_feat['num_model_arr'] == best_params[modeltype_ind][0])[0]
        ind2 = np.where(all_feat['num_component_arr'] == best_params[modeltype_ind][1])[0]
        ind3 = np.where((abs(all_feat['A_arr'] - best_params[modeltype_ind][2]) < 1e-4) & (abs(all_feat['B_arr'] - best_params[modeltype_ind][3]) < 1e-4))[0]

        metric_1 = alpha*all_feat['test_accs'][modeltype_ind, :, :, :, :] + \
                (beta)*all_feat['test_early'][modeltype_ind, :, :, :, :] + \
                (1 - alpha - beta)*all_feat['thresh_met'][modeltype_ind, :, :, :, :]
        metric_1 = metric_1[:, ind1, ind2, ind3]

        data_to_plot = [metric_1.flatten()]
        # print(np.min(metric_1))
        for col, result in zip(col_names[1:], results):
            ind1 = np.where(result['num_model_arr'] == best_params[modeltype_ind][0])[0]
            ind2 = np.where(result['num_component_arr'] == best_params[modeltype_ind][1])[0]
            ind3 = np.where((abs(result['A_arr'] - best_params[modeltype_ind][2]) < 1e-4) & (abs(result['B_arr'] - best_params[modeltype_ind][3]) < 1e-4))[0]

            metric_2 = alpha*result['test_accs'][modeltype_ind, :, :, :, :] + \
                    (beta)*result['test_early'][modeltype_ind, :, :, :, :] + \
                    (1 - alpha - beta)*result['thresh_met'][modeltype_ind, :, :, :, :]
            metric_2 = metric_2[:, ind1, ind2, ind3]

            p = metric_1 - metric_2
            pbar = np.mean(p)
            den = np.sum((p-pbar)**2)/(k-1)
            if den < 1e-6:
                tstat = 0.0
            else:
                tstat = pbar*np.sqrt(k) / np.sqrt(den)
            # interpret via critical value
            if abs(tstat) <= cv1:
                print('\t Removing {} is NOT significant'.format(col))
            else:
                if abs(tstat) <= cv2_1:
                    print('\t Removing {} 0.05 level'.format(col))
                else:
                    print('\t Removing {} 0.01 level'.format(col))

            data_to_plot.append(metric_2.flatten())
            # print(np.min(metric_2))
        axs[modeltype_ind].boxplot(data_to_plot)
        axs[modeltype_ind].set_title(legend[modeltype_ind])
        axs[modeltype_ind].set_xticklabels(col_names)
        axs[modeltype_ind].set_xlabel('Feature Removed')
        axs[modeltype_ind].set_ylim([0.5, 1])
        axs[modeltype_ind].set_ylabel('S')
    plt.show()

def get_fractions():
    filename = 'dataset2_train.pkl'
    with open(filename, 'rb') as f:
        data2 = pickle.load(f)
    print(data2['feat_names'])

    backbutton2 = np.empty((len(data2['T'])))
    feedback2 = np.empty((len(data2['T'])))

    for ii, (y1, y2) in enumerate(zip(data2['Y1'], data2['Y2'])):
        backbutton2[ii] = y2[-1]
        feedback2[ii] = y1[-1]

    val, backbutton2_count = np.unique(backbutton2, return_counts=True)

    val, feedback2_count = np.unique(feedback2, return_counts=True)

    print('Dataset 2 - {} Activities'.format(backbutton2.shape[0]))
    print('Negative {} ({:.1%})'.format(feedback2_count[0], feedback2_count[0]/feedback2.shape[0]))
    print('Neutral {} ({:.1%})'.format(feedback2_count[1], feedback2_count[1]/feedback2.shape[0]))
    print('Positive {} ({:.1%})'.format(feedback2_count[2], feedback2_count[2]/feedback2.shape[0]))

    print('No Backbutton {} ({:.1%})'.format(backbutton2_count[0], backbutton2_count[0]/backbutton2.shape[0]))
    print('Backbutton {} ({:.1%})'.format(backbutton2_count[1], backbutton2_count[1]/backbutton2.shape[0]))

def thresholds():
    alpha = 0.7
    legend = ['Feedback', 'Backbutton']
    best_params = [[1, 6, 0.65, 0.0], [2, 6, 0.50, 0.0]]
    with open('result//result_alpha_' + str(alpha) + '_features_all.pkl', 'rb') as f:
        all_feat = pickle.load(f)

    for modeltype_ind in range(2):
        print(legend[modeltype_ind])
        ind1 = np.where(all_feat['num_model_arr'] == best_params[modeltype_ind][0])[0]
        ind2 = np.where(all_feat['num_component_arr'] == best_params[modeltype_ind][1])[0]
        for a in np.arange(0.55, 1.0, 0.05):
            ind3 = np.where((abs(all_feat['A_arr'] - a) < 1e-4) & (abs(all_feat['B_arr'] - best_params[modeltype_ind][3]) < 1e-4))[0]
            acc = all_feat['test_accs'][modeltype_ind, :, ind1, ind2, ind3]
            early = all_feat['test_early'][modeltype_ind, :, ind1, ind2, ind3]
            freq = all_feat['thresh_met'][modeltype_ind, :, ind1, ind2, ind3]
            print('Threshold {:.2f}, Accuracy {:.3f}, Earliness {:.3f}, Frequency {:.3f}'.format(a, np.mean(acc), np.mean(early), np.mean(freq)))

thresholds()
