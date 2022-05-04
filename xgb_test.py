# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:51:06 2021

@author: gaokaifeng
"""

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold #交叉验证
from time import time
from sklearn.metrics import mean_squared_error


def loadTrainData():
    traindata = pd.read_excel(r'qumian11525data.xlsx')
    train_xy = traindata.loc[:,['X','Y']].values
    train_z = traindata.loc[:,['Z']].values
    return train_xy, train_z

def loadTestData():
    testdata = pd.read_excel(r'qumian11525pred.xlsx')
    test_xy = testdata.loc[:,['X','Y']].values
    test_z = testdata.loc[:,['Z']].values
    return test_xy, test_z

def trainSearch(train_xy, train_z, test_xy):
    print("Parameter optimization")
#    n_estimators = [50, 100, 200, 400]
#    max_depth = [2, 4, 6, 8]
#    learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    n_estimators = [500, 800, 1000, 1300]
    max_depth = [8, 10, 11, 12]
    learning_rate = [0.01, 0.03, 0.05, 0.1]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
    xgb_model = xgb.XGBRegressor()
    kfold = TimeSeriesSplit(n_splits=4)
    grid_search = GridSearchCV(xgb_model, param_grid, verbose=1, scoring = 'neg_mean_squared_error', cv= kfold)
    grid_result = grid_search.fit(train_xy, train_z)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return means, grid_result


def trainandTest(train_xy, train_z, test_xy, test_z):
    # XGBoost训练过程 
    t1 = time()
    model = xgb.XGBRegressor(max_depth=10, learning_rate=0.01, n_estimators=1300)
    model.fit(train_xy, train_z)
    # 对测试集进行预测
    ans = model.predict(test_xy)
    t2 = time()
    print("RMSE: " + str(np.sqrt(mean_squared_error(ans, test_z))))
    t = t2 - t1
    print("Time consumption: {0} s.".format(t))
    
#    #=======================================================
#    feature_importance = model.feature_importances_
#    feature_importance = 100.0 * (feature_importance / feature_importance.max())
#    #print('特征：', train_xy.columns)
#    print('每个特征的重要性：', feature_importance)
#    #sorted_idx = np.argsort(feature_importance)
#    #pos = np.arange(sorted_idx.shape[0])
#    #plt.barh(pos, feature_importance[sorted_idx], align='center')
#    #plt.yticks(pos, train_xy.columns[sorted_idx])
#    #plt.xlabel('Features')
#    #plt.ylabel('Importance')
#    #plt.title('Variable Importance')
#    #plt.show()
#    #=======================================================
    
    #==============================================
    #预测结果输出
    ans_len = len(ans)
    id_list = np.arange(1, 6700)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)
    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'Z_pred'])
    # print(pd_data)
    pd_data.to_csv('qumian11525_result_xgb.csv', index=None)
    #==============================================


if __name__ == '__main__':
#    trainFilePath = r'F:\gaokaifeng\Demo_practice\renormdata.xlsx'
#    testFilePath = r'F:\gaokaifeng\Demo_practice\renormpred.xlsx'
    train_xy, train_z = loadTrainData()
    test_xy, test_z = loadTestData()
#    trainSearch(train_xy, train_z, test_xy)
    trainandTest(train_xy, train_z, test_xy, test_z)








