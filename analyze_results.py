import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper import class_name_from_ind
from model_training import load_model
import os
from mpl_toolkits.axes_grid1 import ImageGrid

def incr_plot():
    headers = np.linspace(0.005, 0.5, 15)

    results = []
    legend = []
    for header in headers:
        results.append(pickle.load(open('result_dataset1_vid_folds_4_model_num_4_incr_{}.pkl'.format(header), 'rb')))
        legend.append(str(header)[:5])
    thresh_arr = results[0]['thresh_arr']

    num_results = len(results)
    num_thresh = thresh_arr.shape[0]
    test_accs = np.empty((2, num_results, num_thresh))
    for model_ind in range(2):
        for res_ind, res in enumerate(results):
            test_accs[model_ind, res_ind, :] = np.mean(np.mean(res['test_accs'][model_ind], 0), 1)

    colors = plt.cm.jet(np.linspace(0.1,0.9,num_results))

    fig, axs = plt.subplots(1,2, sharex=True, figsize=(12,9))
    for ii in range(num_results):
        axs[0].plot(thresh_arr, test_accs[0,ii,:], color=colors[ii])
        axs[1].plot(thresh_arr, test_accs[1,ii,:], color=colors[ii])
    axs[0].legend(legend)
    axs[0].set_ylabel('Average Test Accuracy')
    axs[1].set_ylabel('Average Test Accuracy')
    axs[0].set_xlabel('Threshold')
    axs[1].set_xlabel('Threshold')
    axs[0].set_title('Feedback')
    axs[1].set_title('Backbutton')
    axs[0].set_ylim([0.6, 0.85])
    axs[1].set_ylim([0.6, 0.85])
    plt.show()


def num_model_plot():
    headers = np.arange(1,7)

    results = []
    legend = []
    for header in headers:
        results.append(pickle.load(open('result_dataset1_vid_folds_4_model_num_{}_incr_0.05.pkl'.format(header), 'rb')))
        legend.append(int(header))
    thresh_arr = results[0]['thresh_arr']

    num_results = len(results)
    num_thresh = thresh_arr.shape[0]
    test_accs = np.empty((2, num_results, num_thresh))
    for model_ind in range(2):
        for res_ind, res in enumerate(results):
            test_accs[model_ind, res_ind, :] = np.mean(np.mean(res['test_accs'][model_ind], 0), 1)

    colors = plt.cm.jet(np.linspace(0.1,0.9,num_results))

    fig, axs = plt.subplots(1,2, sharex=True, figsize=(12,9))
    for ii in range(num_results):
        axs[0].plot(thresh_arr, test_accs[0,ii,:], color=colors[ii])
        axs[1].plot(thresh_arr, test_accs[1,ii,:], color=colors[ii])
    axs[1].legend(legend)
    axs[0].set_ylabel('Average Test Accuracy')
    axs[1].set_ylabel('Average Test Accuracy')
    axs[0].set_xlabel('Threshold')
    axs[1].set_xlabel('Threshold')
    axs[0].set_title('Feedback')
    axs[1].set_title('Backbutton')
    axs[0].set_ylim([0.5, 0.85])
    axs[1].set_ylim([0.5, 0.85])
    plt.show()

def num_model_iter_plot():
    filename = 'result_dataset1_vid_folds_4_model_num_4_incr_0.05.pkl'
    base_result = pickle.load(open(filename, 'rb'))
    thresh_arr = base_result['thresh_arr']

    filename = 'result_dataset1_vid_folds_4_model_num_{}_incr_{}.pkl'
    model_num_arr = np.arange(1,7)
    iter_arr = np.linspace(0.005, 0.5, 15)

    feedback_results = np.empty((model_num_arr.shape[0], iter_arr.shape[0], thresh_arr.shape[0]))
    backbutton_results = np.empty((model_num_arr.shape[0], iter_arr.shape[0], thresh_arr.shape[0]))

    for model_ind, model_num in enumerate(model_num_arr):
        for iter_ind, iter in enumerate(iter_arr):
            with open(filename.format(model_num, iter), 'rb') as f:
                data = pickle.load(f)
                feedback_results[model_ind, iter_ind, :] = np.mean(np.mean(data['test_accs'][0], 0), 1)
                backbutton_results[model_ind, iter_ind, :] = np.mean(np.mean(data['test_accs'][1], 0), 1)

    # vmin = np.min(backbutton_results[:, :, :-1])
    # vmax = np.max(backbutton_results[:, :, :-1])
    #
    # fig = plt.figure()
    # grid = ImageGrid(fig, 111, nrows_ncols=(1, len(model_num_arr)), axes_pad=0.1,
    #                 share_all=True,
    #                 cbar_location="right",
    #                 cbar_mode="single",
    #                 cbar_size="7%",
    #                 cbar_pad=0.15,)
    # for ax, model_ind in zip(grid, range(len(model_num_arr))):
    #     acc = backbutton_results[model_ind, :, :-1]
    #     im = ax.imshow(acc, cmap='jet', vmin=vmin, vmax=vmax)
    #     ax.set_xticks(range(thresh_arr.shape[0]-1))
    #     ax.set_xticklabels(np.round(thresh_arr[:-1], decimals=2), rotation=90)
    #     ax.set_yticks(range(iter_arr.shape[0]))
    #     ax.set_yticklabels(np.round(iter_arr, decimals=4))
    #     ax.set_xlabel('Threshold')
    #     ax.set_title('{} Models'.format(model_num_arr[model_ind]))
    # grid[0].set_ylabel('Incr')
    # ax.cax.colorbar(im)
    # ax.cax.toggle_label(True)

    vmin = np.min(feedback_results[:, :, :-1])
    vmax = np.max(feedback_results[:, :, :-1])

    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(model_num_arr)), axes_pad=0.1,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,)
    for ax, model_ind in zip(grid, range(len(model_num_arr))):
        acc = feedback_results[model_ind, :, :-1]
        im = ax.imshow(acc, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(thresh_arr.shape[0]-1))
        ax.set_xticklabels(np.round(thresh_arr[:-1], decimals=2), rotation=90)
        ax.set_yticks(range(iter_arr.shape[0]))
        ax.set_yticklabels(np.round(iter_arr, decimals=4))
        ax.set_xlabel('Threshold')
        ax.set_title('{} Models'.format(model_num_arr[model_ind]))
    grid[0].set_ylabel('Incr')
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    plt.show()
    # fig, axs = plt.subplots(1, len(model_num_arr), sharey=True)
    # for model_ind, model_num in enumerate(model_num_arr):
    #     acc = backbutton_results[model_ind, :, :-1]
    #     im = axs[model_ind].imshow(acc, cmap='jet', vmin=vmin, vmax=vmax)
    #     axs[model_ind].set_xticks(range(thresh_arr.shape[0]-1))
    #     axs[model_ind].set_xticklabels(np.round(thresh_arr[:-1], decimals=2), rotation=90)
    #     axs[model_ind].set_yticks(range(iter_arr.shape[0]))
    #     axs[model_ind].set_yticklabels(np.round(iter_arr, decimals=4))
    #     axs[model_ind].set_xlabel('Threshold')
    #     axs[model_ind].set_title('{} Models'.format(model_num))
    # axs[0].set_ylabel('Iter')
    #
    # fig.colorbar(im, ax=axs.ravel().tolist())
    #
    # plt.show()


def one_plot():
    filename = 'result_dataset1_vid_folds_4_model_num_4_incr_0.05.pkl'
    filename = 'result_dataset2_folds_10_model_num_4_incr_0.05.pkl'
    base_result = pickle.load(open(filename, 'rb'))
    thresh_arr = base_result['thresh_arr']

    test_accs = []
    fracs = []
    for acc, frac in zip(base_result['test_accs'], base_result['test_thresh_not_reached']):
        test_accs.append(np.mean(np.mean(acc, 0), 1))
        frac = np.mean(frac,0)
        fracs.append(frac[:,0]/frac[:,1])

    fig, axs = plt.subplots(2,1, figsize=(6,8))
    axs[0].plot(thresh_arr, test_accs[0], label='Feedback')
    axs[0].plot(thresh_arr, test_accs[1], label='BackButton')
    axs[0].set_ylabel('Test Accuracy of Threshold Met')
    #axs[0].set_xlabel('Threshold')
    axs[0].set_ylim([0, 1])

    axs[1].plot(thresh_arr, 1 - fracs[0], label='Feedback')
    axs[1].plot(thresh_arr, 1 - fracs[1], label='BackButton')
    axs[1].set_ylabel('Fraction of Threshold Met')
    axs[1].set_xlabel('Threshold')
    axs[1].set_ylim([0, 1])
    axs[1].legend()

    plt.show()

def two_plot():
    filename1 = 'result_dataset1_vid_folds_4_model_num_4_incr_0.05_prev.pkl'
    filename2 = 'result_dataset1_vid_folds_4_model_num_4_incr_0.05.pkl'
    result1 = pickle.load(open(filename1, 'rb'))
    result2 = pickle.load(open(filename2, 'rb'))
    thresh_arr = result1['thresh_arr']

    test_accs1 = []
    fracs1 = []
    for acc, frac in zip(result1['test_accs'], result1['test_thresh_not_reached']):
        test_accs1.append(np.mean(np.mean(acc, 0), 1))
        frac = np.mean(frac,0)
        fracs1.append(frac[:,0]/frac[:,1])

    test_accs2 = []
    fracs2 = []
    for acc, frac in zip(result2['test_accs'], result2['test_thresh_not_reached']):
        test_accs2.append(np.mean(np.mean(acc, 0), 1))
        frac = np.mean(frac,0)
        fracs2.append(frac[:,0]/frac[:,1])


    fig, axs = plt.subplots(2,2)
    axs[0, 0].plot(thresh_arr, test_accs1[0], color='r', label='Previous Method')
    axs[0, 0].plot(thresh_arr, test_accs2[0], color='g', label='New Method')
    axs[0, 0].set_ylabel('Test Accuracy of Threshold Met')
    axs[0, 0].set_xlabel('Threshold')
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_title('Feedback')

    axs[0, 1].plot(thresh_arr, test_accs1[1], color='r', label='Previous Method')
    axs[0, 1].plot(thresh_arr, test_accs2[1], color='g', label='New Method')
    axs[0, 1].set_ylabel('Test Accuracy of Threshold Met')
    axs[0, 1].set_xlabel('Threshold')
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].set_title('Backbutton')

    axs[1, 0].plot(thresh_arr, 1-fracs1[0], color='r', label='Previous Method')
    axs[1, 0].plot(thresh_arr, 1-fracs2[0], color='g', label='New Method')
    axs[1, 0].set_ylabel('Fraction of Threshold Met')
    axs[1, 0].set_xlabel('Threshold')
    axs[1, 0].set_ylim([0, 1])
    axs[1, 0].legend()

    axs[1, 1].plot(thresh_arr, 1-fracs1[1], color='r', label='Previous Method')
    axs[1, 1].plot(thresh_arr, 1-fracs2[1], color='g', label='New Method')
    axs[1, 1].set_ylabel('Fraction of Threshold Met')
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].set_ylim([0, 1])
    axs[1, 1].legend()

    plt.show()

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

        counts, bins = np.histogram(times, bins=10, range=(0, 1250))

        axs[ii].bar(bins[:-1], counts/feedback.shape[0], width=(bins[1]-bins[0])*3/4, color=colors[ii])
        axs[ii].set_ylabel('Percentage of Activities')
        axs[ii].set_title(legend2[ii])
        axs[ii].set_ylim([0, 1.0])
        axs[ii].set_xlim([-100, 1250])
    axs[2].set_xlabel('Time (s)')
    plt.show()

def get_fractions():
    filename = 'dataset1_vid.pkl'
    with open(filename, 'rb') as f:
        data1 = pickle.load(f)
    filename = 'dataset2.pkl'
    with open(filename, 'rb') as f:
        data2 = pickle.load(f)

    backbutton1 = np.empty((len(data1['T'])))
    feedback1 = np.empty((len(data1['T'])))
    backbutton2 = np.empty((len(data2['T'])))
    feedback2 = np.empty((len(data2['T'])))

    for ii, (y1, y2) in enumerate(zip(data1['Y1'], data1['Y2'])):
        backbutton1[ii] = y2[-1]
        feedback1[ii] = y1[-1]
    for ii, (y1, y2) in enumerate(zip(data2['Y1'], data2['Y2'])):
        backbutton2[ii] = y2[-1]
        feedback2[ii] = y1[-1]

    val, backbutton1_count = np.unique(backbutton1, return_counts=True)
    val, backbutton2_count = np.unique(backbutton2, return_counts=True)
    val, feedback1_count = np.unique(feedback1, return_counts=True)
    val, feedback2_count = np.unique(feedback2, return_counts=True)

    print('Dataset 1 - {} Activities'.format(backbutton1.shape[0]))
    print('Negative {} ({:.1%})'.format(feedback1_count[0], feedback1_count[0]/feedback1.shape[0]))
    print('Neutral {} ({:.1%})'.format(feedback1_count[1], feedback1_count[1]/feedback1.shape[0]))
    print('Positive {} ({:.1%})'.format(feedback1_count[2], feedback1_count[2]/feedback1.shape[0]))

    print('No Backbutton {} ({:.1%})'.format(backbutton1_count[0], backbutton1_count[0]/backbutton1.shape[0]))
    print('Backbutton {} ({:.1%})'.format(backbutton1_count[1], backbutton1_count[1]/backbutton1.shape[0]))

    print('Dataset 2 - {} Activities'.format(backbutton2.shape[0]))
    print('Negative {} ({:.1%})'.format(feedback2_count[0], backbutton2_count[0]/backbutton2.shape[0]))
    print('Neutral {} ({:.1%})'.format(feedback2_count[1], feedback2_count[1]/feedback2.shape[0]))
    print('Positive {} ({:.1%})'.format(feedback2_count[2], feedback2_count[2]/feedback2.shape[0]))

    print('No Backbutton {} ({:.1%})'.format(backbutton2_count[0], backbutton2_count[0]/backbutton2.shape[0]))
    print('Backbutton {} ({:.1%})'.format(backbutton2_count[1], backbutton2_count[1]/backbutton2.shape[0]))

two_plot()
