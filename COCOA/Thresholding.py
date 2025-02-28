import numpy as np
import pandas as pd

def Thresholding(pre, method, Y):
    """
    parameter: 
    - pre: prediction values
    - method: thresholding method
    - Y: training true labels
    return: 
    - pred: the thresholded prediction values
    """
    if method['type']=='':
        method['type'] = 'Scut'

    numNt, numL = pre.shape
    pred = np.zeros_like(pre)

    # Thresholding logic
    if method['type'].lower() == 'scut':
        # Scut (Single threshold for all labels)
        if 'param' not in method:
            print("Warning: threshold epsilon is not set. Defaulting to 0.5.")
            method['param'] = 0.5
        # Apply thresholding
        pred = (pre > method['param']).astype(int)

    elif method['type'].lower() == 'rcut':
        # Rcut (Rank-based thresholding)
        if 'param' not in method:
            if Y is None:
                raise ValueError("Y (true labels) is required for Rcut if 'param' is not provided.")
            print("Warning: rank threshold not set. Using label cardinality.")
            tmp = np.sum(Y, axis=1)
            LC = int(np.ceil(np.mean(tmp)))
            method['param'] = LC
        # Rank and threshold
        ranks = np.argsort(-pre, axis=1)  # Descending order
        for i in range(numNt):
            pred[i, ranks[i, :method['param']]] = 1

    elif method['type'].lower() == 'pcut':
        # Pcut (Proportion-based thresholding)
        if 'param' not in method:
            if Y is None:
                raise ValueError("Y (true labels) is required for Pcut if 'param' is not provided.")
            print("Warning: proportion scores not set. Using label cardinality.")
            tmp = np.sum(Y, axis=0) / Y.shape[0]
            method['param'] = np.ceil(tmp * numNt).astype(int)
        if len(method['param']) != numL:
            raise ValueError("Proportion score must be defined for each label.")
        # Rank and threshold
        ranks = np.argsort(-pre, axis=0)  # Descending order for each label
        for i in range(numL):
            pred[ranks[:method['param'][i], i], i] = 1

    else:
        # Unsupported method
        raise ValueError(f"{method['type']} is not supported.")

    # Handle null predictions (improve ridge regression and k-NN performance)
    if Y is not None and not np.any(np.sum(Y, axis=1) == 0):
        idzero = np.where(np.sum(pred, axis=1) == 0)[0]
        valmax = np.max(pre[idzero, :], axis=1)
        idmax = np.argmax(pre[idzero, :], axis=1)
        valid = valmax != 0
        pred[idzero[valid], idmax[valid]] = 1

    return pred
