import numpy as np
import pandas as pd
import time


def LP_train(X, Y, classifier):
    """
    parameters:
    - X: training features
    - Y: training labels
    - classifier: training classifier
    return: 
    - model: trained model
    - time_elapsed: training time
    """
    model = []
    Labelset, newY = np.unique(Y, axis=0, return_inverse=True)

    Start_time = time.time()
    model.append(classifier().fit(X, newY))
    time_elapsed = time.time() - Start_time
    model.append(Labelset)

    return model, time_elapsed


def LP_test(X, model):
    """
    parameters:
    - X: test features
    - model: trained model
    return: 
    - pred: predicted labels
    - score: probability of predicted labels
    - time_elapsed: prediction time
    """
    Start_time = time.time()
    pre = model['C'].predict(X)
    score = model['C'].predict_proba(X)
    time_elapsed = time.time() - Start_time

    # Convert predicted label to original label
    pred = model['label_y'][pre, :]
    
    return pred, score, time_elapsed
