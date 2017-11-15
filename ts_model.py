__author__ = 'winstarwang'

import numpy as np
import pandas as pd
import time
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def train_model(model, X_train, y_train):
    ''' 用训练集训练模型 '''

    # 开始计时，训练模型，然后停止计时
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))
    
    return "{:.4f}".format(end-start)

    
def predict_target(model, features, target):
    ''' 用训练好的模型做预测并输出分值'''

    # 开始计时，作出预测，然后停止计时
    start = time.time()
    y_pred = model.predict(features)
    end = time.time()
    
    Xy = pd.DataFrame(features,copy=True)
    Xy['true'] = target
    Xy['pred'] = y_pred
    Xy['residuals'] = Xy['true'] - Xy['pred'] 
    # 输出并返回结果
    run_time = "{:.4f}".format(end - start)
    score = "{:.4f}".format(model.score(features,target))
    print(" Made predictions in {} seconds.\n score is: {}".format(run_time,score))
    return Xy


def train_predict(model, X_train, y_train, X_test, y_test):
    ''' 用一个训练和预测，并输出分值 '''
    
    li = []
    # 输出模型名称和训练集大小
    print("Training a {} using a training set size of {}. . .".format(model.__class__.__name__, len(X_train)))
    li.append(len(X_train))
    # 训练一个分类器
    run_time = train_model(model, X_train, y_train)
    li.append(run_time)
    
    # 输出训练和测试的预测结果
    print("Result for X_train:")
    pred_time_train,score_train = predict_target(model,X_train,y_train)
    print("Result for X_test:")
    pred_time_test,score_test = predict_target(model,X_test,y_test)
    
    li.append(pred_time_test)
    li.append(score_train)
    li.append(score_test)
    
    return li

def measure_result(y_true,y_pred,desc=''):
    
    perf = {}
    perf['desc'] = desc
    perf['mae'] = mean_absolute_error(y_true,y_pred)
    perf['mse'] = mean_squared_error(y_true,y_pred)
    perf['rmse'] = np.sqrt(mean_squared_error(y_true,y_pred))
    perf['r2'] = r2_score(y_true,y_pred)
    
    return perf

def make_df_perf():
    
    df_perf = pd.DataFrame(columns=['desc','r2','mae','rmse','mse'])

    return df_perf

def record_measure_result(df_perf,Xy,desc=''):
    
    df_perf = df_perf.append(measure_result(Xy['true'],Xy['pred'],desc),ignore_index=True)

    return df_perf