import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper import class_name_from_ind
from model_training import load_model
import os
from mpl_toolkits.axes_grid1 import ImageGrid

def feedback_dist_by_backbutton():
    filename = 'dataset2.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    Y1, Y2, T, feat_names = data['Y1'], data['Y2'], data['T'], data['feat_names']

    total_times = np.empty((len(T)))
    backbuttons = np.empty((len(T)))
    feedbacks = np.empty((len(T)))
    for ii, (y1, y2, t) in enumerate(zip(Y1, Y2, T)):
        total_times[ii] = t[-1]
        backbuttons[ii] = y2[-1]
        feedbacks[ii] = y1[-1]

    fig, axs = plt.subplots(3, 2, sharex=True, sharey = True, figsize=(8,8))
    legend1 = ['No Backbutton', 'Backbutton']
    legend2 = ['Negative', 'Neutral', 'Positive']
    colors = ['r', 'y', 'g']

    for ii in [0,1]:
        back = np.where(backbuttons == ii)[0]
        feedback = feedbacks[back]
        times = total_times[back]

        for jj in [0, 1, 2]:
            cur_feed = np.where(feedback == jj)[0]
            cur_time = total_times[cur_feed]
            counts, bins = np.histogram(cur_time, bins=10, range=(0, 1250))
            axs[jj, ii].bar(bins[:-1], counts/cur_feed.shape[0], width=(bins[1]-bins[0])*3/4, color=colors[jj])
            axs[jj, ii].set_title('{} - {}'.format(legend1[ii], legend2[jj]))
            axs[jj, 0].set_ylabel('Percentage of Activities')
            axs[jj, ii].set_ylim([0, 1.0])
            axs[jj, ii].set_xlim([-100, 1250])
        axs[2, ii].set_xlabel('Time (s)')
    plt.show()

def dist_by_backbutton():
    filename = 'dataset1_vid.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    Y1, Y2, T, feat_names = data['Y1'], data['Y2'], data['T'], data['feat_names']

    total_times = np.empty((len(T)))
    backbuttons = np.empty((len(T)))
    feedbacks = np.empty((len(T)))
    for ii, (y1, y2, t) in enumerate(zip(Y1, Y2, T)):
        total_times[ii] = t[-1]
        backbuttons[ii] = y2[-1]
        feedbacks[ii] = y1[-1]

    fig, axs = plt.subplots(1, 2, sharex=True, sharey = True)
    legend1 = ['No Backbutton', 'Backbutton']
    legend2 = ['Negative', 'Neutral', 'Positive']
    colors = ['r', 'y', 'g']

    for ii in [0,1]:
        back = np.where(backbuttons == ii)[0]

        times = total_times[back]
        counts, bins = np.histogram(times, bins=10, range=(0, 90))
        axs[ii].bar(bins[:-1], counts, width=(bins[1]-bins[0])*3/4)
        axs[ii].set_title('{}'.format(legend1[ii]))
        axs[ii].set_ylabel('Activities Count')
        axs[ii].set_xlim([-10, 90+10])
        axs[ii].set_xlabel('Time (s)')

    plt.show()

def feedback_dist_by_time():
    filename = 'dataset2.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    Y1, Y2, T, feat_names = data['Y1'], data['Y2'], data['T'], data['feat_names']

    total_times = np.empty((len(T)))
    backbuttons = np.empty((len(T)))
    feedbacks = np.empty((len(T)))
    for ii, (y1, y2, t) in enumerate(zip(Y1, Y2, T)):
        total_times[ii] = t[-1]
        backbuttons[ii] = y2[-1]
        feedbacks[ii] = y1[-1]

    fig, axs = plt.subplots(3, 1, sharex=True, sharey = True, figsize=(6,7))
    legend1 = ['No Backbutton', 'Backbutton']
    legend2 = ['Negative', 'Neutral', 'Positive']
    colors = ['r', 'y', 'g']

    for ii in [0, 1, 2]:
        feedback = np.where(feedbacks==ii)[0]
        times = total_times[feedback]

        counts, bins = np.histogram(times, bins=5, range=(0, 120))

        axs[ii].bar(bins[:-1], counts/feedback.shape[0], width=(bins[1]-bins[0])*3/4, color=colors[ii])
        axs[ii].set_ylabel('Percentage of Activities')
        axs[ii].set_title(legend2[ii])
        axs[ii].set_ylim([0, 1.0])
        axs[ii].set_xlim([-100, 1250])
    axs[2].set_xlabel('Time (s)')
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

get_fractions()
