import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper import class_name_from_ind
from model_training import load_model
import os

model_split = np.array([[0, 30, 100, 300, 450, -1]])
foldername = 'model_dataset1_vid_RandomForest_1'
importances = np.zeros((4, 5, 8))

for model_ind in range(len(model_split)):
    #For each fold
    for fold_ind in range(4):
        #For each model in model split
        cur_model_arr = [[],[]]
        for model_start_ind in range(len(model_split[model_ind])-1):

            #Load the models into the area
            start_val = model_split[model_ind][model_start_ind]
            end_val = model_split[model_ind][model_start_ind+1]
            model_name = 'Model_{}_Start_{}_End_{}_Fold_{}.joblib'.format(model_ind,
                                start_val, end_val, fold_ind)

            model = load_model(model_name, foldername)

            importances[fold_ind, model_start_ind, :] = model.feature_importances_

importances = np.mean(importances, axis=0)

x = ['0-30 sec', '30-100 sec', '100-300 sec', '300-450 sec', '450-end sec']
feature_names = ['Head Proximity', 'Head Orientation', 'Gaze Direction', 'Eye Aspect Ratio', 'Pupil Ratio', 'Progress', 'Picture Side', 'Activity Type']

fig, ax = plt.subplots()
for ind in range(8):
    ax.plot(x, importances[:,ind], label=feature_names[ind])

ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Importances')
ax.set_title('')
plt.show()

dec_tree = pickle.load(open('result_dataset1_vid_DecisionTree_1.pkl', 'rb'))
rand_forest = pickle.load(open('result_dataset1_vid_RandomForest_1.pkl', 'rb'))
svc = pickle.load(open('result_dataset1_vid_SVC_1.pkl', 'rb'))

results = [dec_tree, rand_forest, svc]
legend = ['Decision Tree', 'Random Forest', 'SVC']

train_accs = [[],[]]
test_accs = [[],[]]
for model_ind in range(model_split.shape[0]):
    for res in results:
        train_accs[model_ind].append(res['train_accs'][model_ind])
        test_accs[model_ind].append(res['test_accs'][model_ind])

num_thresh = results[0]['train_accs'][0].shape[1]
thresh_arr = np.linspace(0, 1, num_thresh).astype('float')

#First model
train_accs = np.array(train_accs[0])
test_accs = np.array(test_accs[0])

class_num = 3
fig, ax = plt.subplots(2,3, figsize=(8, 6))
for label in range(class_num):
    for res_ind in range(len(results)):
        ax[0,label].plot(thresh_arr, np.mean(train_accs[res_ind,:,:,label], axis=0).flatten())
        ax[1,label].plot(thresh_arr, np.mean(test_accs[res_ind,:,:,label], axis=0).flatten())

    ax[0,label].set_xlim([0, 1])
    ax[0,label].set_ylim([0, 1])
    ax[0,label].set_title(class_name_from_ind(label, num_classes=class_num))

    ax[1,label].set_xlim([0, 1])
    ax[1,label].set_ylim([0, 1])
    ax[1,label].set_xlabel('Threshold')

ax[0,0].set_ylabel('{} Accuracy'.format('Train'))
ax[0,class_num-1].legend(legend)
ax[1,0].set_ylabel('{} Accuracy'.format('Test'))
ax[1,class_num-1].legend(legend)

plt.show()
