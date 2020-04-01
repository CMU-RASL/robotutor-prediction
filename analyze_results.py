import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper import class_name_from_ind
from model_training import load_model
import os

def max_depth_plot():
    headers = ['1', '6', '11', '17', '22', '28', '33', '39', '44', '50']

    results = []
    legend = []
    for header in headers:
        results.append(pickle.load(open('result_dataset1_vid_RandomForest_1__' + header + '.pkl', 'rb')))
        legend.append(int(header))

    train_accs = [[],[]]
    test_accs = [[],[]]
    for model_ind in range(model_split.shape[0]):
        for res in results:
            test_accs[model_ind].append(res['test_accs'][model_ind])

    num_thresh = results[0]['train_accs'][0].shape[1]
    thresh_arr = np.linspace(1/3, 1, num_thresh).astype('float')

    test_accs = np.array(test_accs[0])
    test_accs = np.mean(np.mean(test_accs, axis=1), axis=2)[:,3:-1]
    plt.plot(legend, test_accs)
    plt.xlabel('Max Depth')
    plt.ylabel('Average Test Accuracy')
    plt.legend(np.round(thresh_arr[3:-1], decimals=2))
    plt.show()

def ablation_plot():
    headers = ['Head Proximity', 'Head Orientation', 'Gaze Direction',
                'Eye Aspect Ratio', 'Pupil Ratio', 'Progress', 'Picture Side',
                'Activity']
    max_depth = '23'
    base_result = pickle.load(open('result_dataset1_vid_RandomForest_1__' + max_depth + '.pkl', 'rb'))

    results = []
    legend = []
    for header in headers:
        results.append(pickle.load(open('result_dataset1_vid_RandomForest_1_' + header.replace(" ","") + '_' + max_depth + '.pkl', 'rb')))
        legend.append(header)

    test_accs = [[],[]]
    for model_ind in range(model_split.shape[0]):
        for res in results:
            test_accs[model_ind].append(res['test_accs'][model_ind] - base_result['test_accs'][model_ind])

    num_thresh = results[0]['train_accs'][0].shape[1]
    thresh_arr = np.linspace(0.5, 0.9, num_thresh).astype('float')

    test_accs = np.array(test_accs[0])
    test_accs = np.mean(np.mean(test_accs, axis=1), axis=2)
    print(thresh_arr[2])
    print(test_accs[:, 2])
    plt.plot(thresh_arr, test_accs.T)
    plt.plot([0, 1], [0, 0], 'k')
    #plt.plot([0.56, 0.56], [-0.03, 0.03], 'k')
    plt.legend(legend)
    plt.xlabel('Threshold')
    plt.ylabel('Test Accuracy')
    plt.xlim([0.4, 1])
    plt.ylim([-0.04, 0.04])
    plt.show()


model_split = np.array([[0, 30, 100, 300, 450, -1]])
# foldername = 'model_dataset1_vid_RandomForest_1_'
# # importances = np.zeros((4, 5, 8))
# #
# for model_ind in range(len(model_split)):
#     #For each fold
#     for fold_ind in range(4):
#         #For each model in model split
#         cur_model_arr = [[],[]]
#         for model_start_ind in range(len(model_split[model_ind])-1):
#
#             #Load the models into the area
#             start_val = model_split[model_ind][model_start_ind]
#             end_val = model_split[model_ind][model_start_ind+1]
#             model_name = 'Model_{}_Start_{}_End_{}_Fold_{}.joblib'.format(model_ind,
#                                 start_val, end_val, fold_ind)
#
#             model = load_model(model_name, foldername)
#             print(np.max([m.get_depth() for m in model.estimators_]))
#
#             print([estimator.get_depth() for estimator in model.estimators_])
#
# importances = np.mean(importances, axis=0)

# x = ['0-30 sec', '30-100 sec', '100-300 sec', '300-450 sec', '450-end sec']
# feature_names = ['Head Proximity', 'Head Orientation', 'Gaze Direction', 'Eye Aspect Ratio', 'Pupil Ratio', 'Progress', 'Picture Side', 'Activity Type']
#
# fig, ax = plt.subplots()
# for ind in range(8):
#     ax.plot(x, importances[:,ind], label=feature_names[ind])
#
# ax.legend()
# ax.set_xlabel('Time')
# ax.set_ylabel('Importances')
# ax.set_title('')
# plt.show()

# headers = ['Head Proximity', 'Head Orientation', 'Gaze Direction',
#             'Eye Aspect Ratio', 'Pupil Ratio', 'Progress', 'Picture Side',
#             'Activity']
# headers = ['1', '6', '11', '17', '22', '28', '33', '39', '44', '50']
# base_result = pickle.load(open('result_dataset1_vid_RandomForest_1_.pkl', 'rb'))
#
# results = []
# legend = []
# for header in headers:
#     results.append(pickle.load(open('result_dataset1_vid_RandomForest_1__' + header + '.pkl', 'rb')))
#     legend.append(int(header))
#
# train_accs = [[],[]]
# test_accs = [[],[]]
# for model_ind in range(model_split.shape[0]):
#     for res in results:
#         train_accs[model_ind].append(res['train_accs'][model_ind])
#         test_accs[model_ind].append(res['test_accs'][model_ind])
#
# num_thresh = results[0]['train_accs'][0].shape[1]
# thresh_arr = np.linspace(1/3, 1, num_thresh).astype('float')
#
# #First model
# train_accs = np.array(train_accs[0])
# test_accs = np.array(test_accs[0])
#
# train_accs = np.mean(np.mean(train_accs, axis=1), axis=2)
# test_accs = np.mean(np.mean(test_accs, axis=1), axis=2)[:,3:-1]
# #print(legend.shape, test_accs.shape)
# plt.plot(legend, test_accs)
# plt.xlabel('Max Depth')
# plt.ylabel('Average Test Accuracy')
# plt.legend(np.round(thresh_arr[3:-1], decimals=2))
# # plt.plot(thresh_arr, test_accs.T)
# # #plt.plot([0, 1], [0, 0], 'k')
# # #plt.plot([0.56, 0.56], [-0.03, 0.03], 'k')
# # plt.legend(legend)
# # plt.xlabel('Threshold')
# # plt.ylabel('Test Accuracy')
# # plt.xlim([0, 1])
# # #plt.ylim([-0.03, 0.03])
# plt.show()
ablation_plot()
