import numpy as np
import pandas as pd
import time


def BR_train(X, Y, classifier, alpha):
    """
    parameters:
    - X: training features
    - Y: training labels
    - classifier: training classifier
    - alpha: regularization parameter
    return: 
    - model: trained model
    - time_elapsed: training time
    """
    n_samples, n_features = X.shape
    _, n_classes = Y.shape
    # classifier = method['param']['base']
    model = []
    
    start_time = time.time()
    if classifier == 'Ridge': 
        if alpha == '':
            alpha = 0.1
        # Calculate the model for each label according to the Ridge formula
        for i in range(0, n_classes):
            # model.append(classifier.fit(X, Y[:, i]))
            XX = np.hstack([np.ones((n_samples, 1)), X])
            I = np.eye(XX.shape[1])     # construct the identity matrix
            tmpinvX = XX.T @ XX + alpha * I  
            invX = np.linalg.solve(tmpinvX, XX.T)
            model.append(invX @ Y[:, i])
    else:
        raise ValueError('classifier is not supported')

    time_elapsed = time.time() - start_time
    
    return model, time_elapsed

def BR_test(Xt, model):
    """
    parameters:
    - Xt: testing features
    - model: trained model
    return: 
    - pre: predicted labels
    - score: predicted scores
    - time_elapsed: testing time
    """
    n_samples, n_features = Xt.shape
    n_classes = len(model)
    pre = np.zeros((Xt.shape[0], n_classes))   # Initialize the predicted label matrix
    score = np.zeros((Xt.shape[0], n_classes)) # Initialize the confidence matrix

    Start_time = time.time()
    # Using the model to make predictions for each label
    for i in range(0, n_classes):
    #     pre[i] = model[i].predict(Xt)
    #     score[i] = model[i].decision_function(Xt)[:,1]
        XXt = np.hstack([np.ones((n_samples, 1)), Xt])
        score[:, i] = XXt @ model[i]
    time_elapsed = time.time() - Start_time

    return pre.T, score, time_elapsed