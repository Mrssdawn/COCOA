import numpy as np
import pandas as pd
import warnings
import json
from scipy.io import loadmat
from functools import partial
from sklearn.svm import SVC
# import sklearn.tree
# from sklearn.tree import DecisionTreeClassifier  # 使用 CART 算法
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from BR_Ridge import BR_train, BR_test
from triClass import triClass_train, triClass_test
from Thresholding import Thresholding


class COCOA:
    def __init__(self,binary_Classifier = 'Ridge', alpha = 10, 
                 multi_classifier = partial(SVC,kernel='linear',C=1,probability=True,verbose=False),
                 K = 3):
        """
        Paper: Towards Class-Imbalance Aware Multi-Label Learning
        COCOA: https://ieeexplore.ieee.org/abstract/document/9262911
        parameters:
        - binary_Classifier: binary classifier
        - alpha: the hyperparameter of binary classifier
        - multi_classifier: multi-classifier
        - K: the number of labels to be selected
        return: 
        - None
        """
        self.K = K
        self.alpha = alpha
        self.binary_classifier = binary_Classifier
        self.multi_classifier = multi_classifier  
        self.model = None  # store model
        self.T_train = 0  # store training time

    def COCOA_train(self, X, Y):
        """
        parameters:
        - X: training features
        - Y: training labels
        return: 
        - None
        """
        n_samples, n_features = X.shape
        _, n_classes = Y.shape

        # BR
        binary_model, binary_time = BR_train(X, Y, self.binary_classifier, self.alpha)

        # triClass
        Multi_model, Multi_time = triClass_train(X, Y, self.multi_classifier, self.K)
        self.model = [binary_model , Multi_model]
        self.T_train = binary_time + Multi_time

    
    def COCOA_test(self, Xt, Y):
        """
        parameters:
        - Xt: test features
        - Yt: test labels
        return: 
        - pre: predicted labels
        - conf: probability of predicted labels
        - all_time: prediction time
        """
        binary_time, Multi_time = 0, 0

        # BR
        _, binary_score, binary_time = BR_test(Xt, self.model[0])

        # triClass
        Multi_pre, Multi_score, Multi_time = triClass_test(Xt, self.model[1])
        conf = (binary_score + Multi_pre)/2

        th = {'type': 'Scut',
              'param':0.65}
        pred = Thresholding(conf, th, Y.T)
        pre = pred.T
        conf = conf.T

        all_time = binary_time + Multi_time

        return pre, conf, all_time
    


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    datas =['flags','foodtruck','CHD_49','emotions','yeast','birds',
                'Image','scene','VirusPseAAC','PlantPseAAC',
                'Enron','GnegativeGO']
    index = datas[3]
    data = loadmat("dataset/"+index+".mat")
    X = data['data']
    Y = data['target'].T

    # 读取 JSON 文件
    # json_filename = f'dataset/{index}_3cv.json'
    # with open(json_filename, 'r') as f:
    #     splits = json.load(f)

    # # 获取某一折的训练和测试索引（例如，读取第1折）
    # fold_idx = 0  # 对应第一折
    # train_idx = splits[fold_idx]['train_indices']
    # test_idx = splits[fold_idx]['test_indices']

    # co = COCOA()
    # Macro_F1, Micro_F1 = co.COCOA(X[train_idx], Y[train_idx], X[test_idx], Y[test_idx])
    # print(Macro_F1, Micro_F1)

    num_samples = X.shape[0]  # 获取样本数量
    num_train_samples = int(0.6 * num_samples)  # 80% 用作训练集

    # 分割数据集为训练集和测试集
    train_data = X[:num_train_samples, :]
    train_target = Y[:num_train_samples, :]
    test_data = X[num_train_samples:, :]
    test_target = Y[num_train_samples:, :]

    co = COCOA()
    co.COCOA_train(train_data, train_target)
    pre, conf, T_test = co.COCOA_test(test_data, train_target)
    T_train = co.T_train
    Macro_F1 = f1_score(test_target.T, pre, average='macro')
    Micro_F1 = f1_score(test_target.T, pre, average='micro')
    Macro_auc = roc_auc_score(test_target.T, conf, average='macro')
    print(Macro_F1, Micro_F1, Macro_auc, T_train, T_test)