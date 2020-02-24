import numpy as np

def class_name_from_ind(ind, num_classes):
    if num_classes == 2:
        if ind == 0:
            return 'Completed'
        else:
            return 'Bailed'
    else:
        if ind == 0:
            return 'Negative'
        elif ind == 1:
            return 'Neutral'
        else:
            return 'Positive'

def choose_model(tt, model_split):
    for ii in range(len(model_split)):
        if tt >= model_split[ii][0] and tt < model_split[ii][1]:
            return ii
    return len(model_split)-1

def get_prob(model, x, num_classes):
    prob = np.zeros(num_classes).astype('float')
    pred = model.predict_proba(x).flatten()
    for ii, class_ind in enumerate(model.classes_):
        prob[int(class_ind)] = pred[ii]
    return prob

def get_metrics(conf_mat, num_classes):
    if num_classes == 2:
        cur_class = 1
        not_cur_class = 0
        tp = conf_mat[cur_class,cur_class].astype('float')
        tn = conf_mat[not_cur_class,not_cur_class].astype('float')
        fp = conf_mat[not_cur_class,cur_class].astype('float')
        fn = conf_mat[cur_class,not_cur_class].astype('float')
        tpr = tp/(tp + fn + 1e-6)
        fpr = fp/(fp + tn + 1e-6)
        acc = (tp + tn)/(tp + tn + fp + fn)
        
    else:
        tpr = np.empty(3)
        fpr = np.empty(3)
        acc = np.empty(3)
        for cur_class in range(num_classes):
            not_cur_class = list(range(num_classes))
            not_cur_class.remove(cur_class)
            tp = conf_mat[cur_class,cur_class].astype('float')
            tn = np.sum(conf_mat[not_cur_class,not_cur_class]).astype('float')
            fp = np.sum(conf_mat[not_cur_class,cur_class]).astype('float')
            fn = np.sum(conf_mat[cur_class,not_cur_class]).astype('float')
            tpr[cur_class] = tp/(tp + fn + 1e-6)
            fpr[cur_class] = fp/(fp + tn + 1e-6)
            acc[cur_class] = (tp + tn)/(tp + tn + fp + fn)
            
    return fpr, tpr, acc