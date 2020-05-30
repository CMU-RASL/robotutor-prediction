import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper import class_name_from_ind
from model_training import load_model
import os
from mpl_toolkits.axes_grid1 import ImageGrid

def hypothesisI():

    with open('result//result_features_all.pkl', 'rb') as f:
        all_feat = pickle.load(f)
    with open('result//result_features_face.pkl', 'rb') as f:
        face_feat = pickle.load(f)
    with open('result//result_features_context.pkl', 'rb') as f:
        context_feat = pickle.load(f)

    #Find best indices
    alpha = 0.7
    k = 10

    cv1 = 1.833 #one-sided t-test with 9 df
    cv2 = 2.262 #two-sided t-test with 9 df
    cv1_1 = 2.821 #one-sided t-test with 9 df
    cv2_1 = 3.250 #two-sided t-test with 9 df
    legend = ['Feedback', 'Backbutton']

    fig, axs = plt.subplots(1, 2, figsize=(9,4))
    for modeltype_ind in range(2):
        print(legend[modeltype_ind])
        ind1, ind2, ind3 = all_feat['best_inds'][modeltype_ind]
        acc_all_feat = all_feat['test_accs'][modeltype_ind, :, ind1, ind2, ind3]
        early_all_feat = all_feat['test_early'][modeltype_ind, :, ind1, ind2, ind3]
        metric_all_feat = alpha*acc_all_feat + (1-alpha)*early_all_feat
        print(np.std(metric_all_feat))
        print(np.std(acc_all_feat))
        print(np.std(early_all_feat))
        inds_face_feat = face_feat['best_inds'][modeltype_ind]
        acc_face_feat = face_feat['test_accs'][modeltype_ind, :, ind1, ind2, ind3]
        early_face_feat = face_feat['test_early'][modeltype_ind, :, ind1, ind2, ind3]
        metric_face_feat = alpha*acc_face_feat + (1-alpha)*early_face_feat

        inds_context_feat = context_feat['best_inds'][modeltype_ind]
        acc_context_feat = context_feat['test_accs'][modeltype_ind, :, ind1, ind2, ind3]
        early_context_feat = context_feat['test_early'][modeltype_ind, :, ind1, ind2, ind3]
        metric_context_feat = alpha*acc_context_feat + (1-alpha)*early_context_feat

        data_to_plot = [metric_face_feat, metric_context_feat, metric_all_feat]

        p = metric_all_feat - metric_face_feat
        pbar = np.mean(p)
        den = np.sum((p-pbar)**2)/(k-1)
        tstat = pbar*np.sqrt(k) / np.sqrt(den)
        # interpret via critical value
        if abs(tstat) <= cv1:
        	print('\tFace and Context + Face have the same performance')
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
        	print('\tContext and Context + Face have the same performance')
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
        	print('\tContext and Face have the same performance')
        else:
            if abs(tstat) <= cv2_1:
                print('\tContext and Face 0.05 level')
            else:
                print('\tContext and Face 0.01 level')

        axs[modeltype_ind].boxplot(data_to_plot)
        axs[modeltype_ind].set_title(legend[modeltype_ind])
        axs[modeltype_ind].set_xticklabels(['Face\nOnly', 'Context\nOnly', 'Face and\nContext'])
        axs[modeltype_ind].set_ylim([0, 1])
        axs[modeltype_ind].set_ylabel('S')
    plt.show()

def hypothesisII():
    with open('result//result_features_all.pkl', 'rb') as f:
        all_feat = pickle.load(f)

    #Find best indices
    alpha = 1.0
    k = 10

    cv0 = 1.383
    cv1 = 1.833 #one-sided t-test with 9 df
    cv2 = 2.262 #two-sided t-test with 9 df
    legend = ['Feedback', 'Backbutton']

    fig, axs = plt.subplots(1, 2, figsize=(6,4))
    inds1 = ((0, 5, 0),(0, 3, 9))
    inds2 = ((1, 5, 8),(1, 5, 6))
    for modeltype_ind in range(2):
        print(legend[modeltype_ind])
        acc_1 = all_feat['test_accs'][modeltype_ind, :, inds1[modeltype_ind][0], inds1[modeltype_ind][1], inds1[modeltype_ind][2]]
        early_1 = all_feat['test_early'][modeltype_ind, :, inds1[modeltype_ind][0], inds1[modeltype_ind][1], inds1[modeltype_ind][2]]
        metric_1 = alpha*acc_1 + (1-alpha)*acc_1

        acc_2 = all_feat['test_accs'][modeltype_ind, :, inds2[modeltype_ind][0], inds2[modeltype_ind][1], inds2[modeltype_ind][2]]
        early_2 = all_feat['test_early'][modeltype_ind, :, inds2[modeltype_ind][0], inds2[modeltype_ind][1], inds2[modeltype_ind][2]]
        metric_2 = alpha*acc_2 + (1-alpha)*acc_2

        p = metric_2 - metric_1
        pbar = np.mean(p)
        den = np.sum((p-pbar)**2)/(k-1)
        tstat = pbar*np.sqrt(k) / np.sqrt(den)
        print(tstat)
        # interpret via critical value
        if abs(tstat) <= cv1:
        	print('\tOne interval and more than one interval have the same performance')
        else:
        	print('\tOne interval and more than one interval - 0.05 level')

        data_to_plot = [metric_1, metric_2]
        axs[modeltype_ind].boxplot(data_to_plot)
        axs[modeltype_ind].set_title(legend[modeltype_ind])
        axs[modeltype_ind].set_xticklabels(['1 Time\nInterval', 'More than 1\nTime Interval'])
        axs[modeltype_ind].set_ylim([0, 1])
        axs[modeltype_ind].set_ylabel('S')
    plt.show()

def sensitivity():
    cols = ['Activity Ind', 'Video Time', 'Head Proximity', 'Head Orientation',
            'Gaze Direction', 'Eye Aspect Ratio', 'Pupil Ratio', 'AU04', 'AU07', 'AU12',
            'AU25', 'AU26', 'AU45', 'Progress', 'Picture Side', 'Activity Type', 'Activity Time']
    col_names = ['None', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
    with open('result//result_features_all.pkl', 'rb') as f:
        all_feat = pickle.load(f)

    results = []
    for col in cols:
        with open('result//result_features_all_remove_'+col+'.pkl', 'rb') as f:
            results.append(pickle.load(f))

    #Find best indices
    alpha = 0.7
    k = 10

    cv1 = 1.833 #one-sided t-test with 9 df
    cv2 = 2.262 #two-sided t-test with 9 df
    cv1_1 = 2.821 #one-sided t-test with 9 df
    cv2_1 = 3.250 #two-sided t-test with 9 df
    legend = ['Feedback', 'Backbutton']
    fig, axs = plt.subplots(1, 2, figsize=(10,6))
    inds_base = ((0, 5, 0), (1, 5, 6))
    inds = ((0, 0, 0), (1, 0, 6))
    for modeltype_ind in range(2):
        print(legend[modeltype_ind])
        acc_1 = all_feat['test_accs'][modeltype_ind, :, inds_base[modeltype_ind][0], inds_base[modeltype_ind][1], inds_base[modeltype_ind][2]]
        early_1 = all_feat['test_early'][modeltype_ind, :, inds_base[modeltype_ind][0], inds_base[modeltype_ind][1], inds_base[modeltype_ind][2]]
        metric_1 = alpha*acc_1 + (1-alpha)*acc_1

        data_to_plot = [metric_1]
        for col, result in zip(col_names[1:], results):
            acc_2 = result['test_accs'][modeltype_ind, :, inds[modeltype_ind][0], inds[modeltype_ind][1], inds[modeltype_ind][2]]
            early_2 = result['test_early'][modeltype_ind, :, inds[modeltype_ind][0], inds[modeltype_ind][1], inds[modeltype_ind][2]]
            metric_2 = alpha*acc_2 + (1-alpha)*acc_2

            p = metric_1 - metric_2
            pbar = np.mean(p)
            den = np.sum((p-pbar)**2)/(k-1)
            tstat = pbar*np.sqrt(k) / np.sqrt(den)
            # interpret via critical value
            if abs(tstat) <= cv1:
            	print('\t Removing {} is NOT significant'.format(col))
            else:
                if abs(tstat) <= cv2_1:
                    print('\t Removing {} 0.05 level'.format(col))
                else:
                    print('\t Removing {} 0.01 level'.format(col))

            data_to_plot.append(metric_2)
        axs[modeltype_ind].boxplot(data_to_plot)
        axs[modeltype_ind].set_title(legend[modeltype_ind])
        axs[modeltype_ind].set_xticklabels(col_names)
        axs[modeltype_ind].set_xlabel('Feature Removed')
        axs[modeltype_ind].set_ylim([0, 1])
        axs[modeltype_ind].set_ylabel('S')
    plt.show()

def get_fractions():
    filename = 'dataset2.pkl'
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

hypothesisI()
