import matplotlib.pyplot as plt
import numpy as np
from helper import class_name_from_ind

def plot_probability(t, prob, legend, title, result, count_text):

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(legend) == 2:
        colors = ['b', 'm']
    else:
        colors = ['r', 'y', 'g']
        
    for ii, label in enumerate(legend):
        plt.plot(t, prob[:,ii], colors[ii], label=label)
    plt.ylim([0, 1])
    plt.title(title, fontsize=20)
    plt.legend(fontsize=18, bbox_to_anchor=(1, 1))
    plt.text(1.05, 0.2, count_text, fontsize=18, transform=ax.transAxes, 
             bbox=dict(boxstyle="round", fc="w"))
    if result == 'Correct':
        plt.text(1.05, 0.0, result, fontsize=20, color="b", 
                 transform=ax.transAxes, bbox=dict(boxstyle="round", 
                                                   alpha=0.2, fc="k"))
    else:
        plt.text(1.05, 0.0, result, fontsize=20, color="m", 
                 transform=ax.transAxes, bbox=dict(boxstyle="round", 
                                                   alpha=0.2, fc="k"))
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_mat, num_classes, legend, name):
    norm_conf_mat = conf_mat / np.sum(conf_mat)
    plt.figure(figsize=(9, 5))
    plt.imshow(conf_mat, cmap='Blues')
    acc = np.trace(conf_mat) / np.sum(conf_mat)
    plt.title('{} Confusion Matrix: Accuracy: {:.2%}'.format(name, acc), fontsize=24)
    plt.xticks(np.arange(num_classes), legend, fontsize=16)
    plt.yticks(np.arange(num_classes), legend, fontsize=16)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    cmap = plt.cm.get_cmap('Greys')
    for ii in range(num_classes):
        for jj in range(num_classes):
            plt.text(jj, ii, str(np.round(conf_mat[ii,jj], decimals=2)),
                     ha='center', color=cmap(1-norm_conf_mat[ii,jj]), 
                     fontsize=24)
    plt.tight_layout()

def plot_rocs(fprs, tprs, thresh_arr, name):
    batch_num, thresh_num, class_num = fprs.shape
    avg_fprs = np.mean(fprs, axis=0)
    avg_tprs = np.mean(fprs, axis=0)
    
    legend = []
    for batch in range(batch_num):
        legend.append('Batch {}'.format(batch+1))
    legend.append('Average')
    fig, ax = plt.subplots(1,class_num, figsize=(10,5))
    
    if class_num > 1:
        for label in range(class_num):
            for batch in range(batch_num):           
                ax[label].plot(fprs[batch,:,label], tprs[batch,:,label])
            ax[label].set_xlim([0, 1])
            ax[label].set_ylim([0, 1])
            ax[label].set_xlabel('False Positive Rate')
            ax[label].plot(avg_fprs[:,label], avg_tprs[:,label],'k')
            ax[label].set_title(class_name_from_ind(label, num_classes=class_num))
        
        ax[0].set_ylabel('{} True Positive Rate'.format(name))
        ax[class_num-1].legend(legend)
    else:
        for batch in range(batch_num):           
            ax.plot(fprs[batch,:,0], tprs[batch,:,0])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate')
        ax.plot(avg_fprs[:,0], avg_tprs[:,0],'k')
        ax.set_title(class_name_from_ind(1, num_classes=2))
        ax.set_ylabel('{} True Positive Rate'.format(name))
        ax.legend(legend)

def plot_accuracies(acc, thresh_arr,name):
    batch_num, thresh_num, class_num = acc.shape
    avg_acc = np.mean(acc, axis=0)

    legend = []
    for batch in range(batch_num):
        legend.append('Batch {}'.format(batch+1))
    legend.append('Average')
    fig, ax = plt.subplots(1,class_num, figsize=(10,5))
    
    if class_num > 1:
        for label in range(class_num):
            for batch in range(batch_num):           
                ax[label].plot(thresh_arr, acc[batch,:,label].flatten())
            ax[label].set_xlim([0, 1])
            ax[label].set_ylim([0, 1])
            ax[label].set_xlabel('Threshold')
            ax[label].plot(thresh_arr, avg_acc[:,label],'k')
            ax[label].set_title(class_name_from_ind(label, num_classes=class_num))
        
        ax[0].set_ylabel('{} Accuracy'.format(name))
        ax[class_num-1].legend(legend)
    else:
        for batch in range(batch_num):           
                ax.plot(thresh_arr, acc[batch,:,0].flatten())
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Threshold')
        ax.plot(thresh_arr, avg_acc,'k')
        ax.set_title(class_name_from_ind(1, num_classes=2))
        ax.set_ylabel('{} Accuracy'.format(name))
        ax.legend(legend)
        
    plt.show()