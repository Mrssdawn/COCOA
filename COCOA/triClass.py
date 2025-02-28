import numpy as np
import pandas as pd
from LP import LP_train, LP_test

def triClass_train(X, Y, classifier, K):
    """
    parameters:
    - X: training features
    - Y: training labels
    - classifier: training classifier
    - K: number of labels to be selected
    return: 
    - model: trained model
    - time_elapsed: training time
    """
    n_samples, n_features = X.shape
    _, n_classes = Y.shape
    model = {
    'C': [],  # store the model
    'label_y': [],  # store the triClass label values
    'label': []  # store the triClass label
    }
    all_time = 0

    for i in range(0, n_classes):
        # Remove the current label from all labels
        candLab = list(set(range(n_classes)) - {i})
    
        if (n_classes - 1) < K:
            K = n_classes - 1

        # Randomly select K labels from the candidate labels
        sampledLab = np.random.choice(candLab, K, replace=False)        
        tmpY = Y[:, [i] + list(sampledLab)]  # Constructing a three-class classification problem
        indIns = (Y[:, i] > 0)  # Get the current label instance      
        tmpY[indIns, 1:] = 0   # Set instances of non-current labels to 0

        model_label, train_time = LP_train(X, tmpY, classifier)
        model['C'].append(model_label[0])
        model['label_y'].append(model_label[1])
        all_time += train_time
        model['label'].append([i] + list(sampledLab))
    
    return model, all_time


def triClass_test(Xt, model):
    """
    parameters:
    - Xt: test features
    - model: trained model
    return: 
    - pred: predicted labels
    - score: probability of predicted labels
    - all_time: prediction time
    """
    n_samples, n_features = Xt.shape
    n_classes = len(model['C'])
    
    pre = np.zeros((Xt.shape[0], n_classes))  # Initialize the predicted label matrix
    score = np.zeros((Xt.shape[0], n_classes)) # Initialize the confidence matrix
    test_time = [None] * n_classes  # Initialize the prediction time

    for label in range(n_classes):
        # Calling the test model for each label
        newmodel = {'C':model['C'][label],
                'label_y':model['label_y'][label]}

        tmppre, tmpscore, test_time[label] = LP_test(Xt, newmodel)
        # Calculate the confidence of the current label based on the probability obtained by question conversion
        indices = np.where(newmodel['label_y'][:, 0] == 1)[0]
        selected_columns = tmpscore[:, indices]
        score[:, label] = np.sum(selected_columns, axis=1)
        # Get the predicted value of the current label
        pre[:, label] = tmppre[:, 0]

    all_time = sum(test_time)
    
    return pre, score, all_time
