# coding: utf-8
import os
import tushare as ts
import pandas as pd
import numpy as np
from math import isnan
import matplotlib.pyplot as plt
import scipy.optimize as sco


def process(df_raw):
    code2nan_num = {}
    for code in df_raw.columns:
        code2nan_num[code] = df_raw[code].isnull().sum()
    nan_info = {'0': 0, '1': 0, '1-10': 0,
                '10-100': 0, '100-1000': 0, '1000+': 0}
    for code, nan_num in code2nan_num.items():
        if nan_num == 0:
            nan_info['0'] += 1
        elif nan_num == 1:
            nan_info['1'] += 1
        elif 1 < nan_num <= 10:
            nan_info['1-10'] += 1
        elif 10 < nan_num <= 100:
            nan_info['10-100'] += 1
        elif 100 < nan_num <= 1000:
            nan_info['100-1000'] += 1
        else:
            nan_info['1000+'] += 1
    print('nan info:')
    for k, v in nan_info.items():
        print(k, ':')
        print(v)
    plt.hist(list(code2nan_num.values()))
    plt.title('nan data distribution')
    valid_codes = []
    invalid_codes = []
    for code, nan_num in code2nan_num.items():
        if nan_num >= 6:
            invalid_codes.append(code)
        else:
            valid_codes.append(code)

    print('删去缺失信息较多的股票，共删去{}只股票'.format(len(invalid_codes)))
    print('筛选后的股票数量:', len(valid_codes))

    valid_df_stocks = df_raw.drop(invalid_codes, axis=1, inplace=False)
    valid_df_stocks = valid_df_stocks.interpolate(
        method='linear', limit_direction='forward', axis=0) #插值会导致数据被污染
    valid_df_stocks = valid_df_stocks.dropna()
    log_valid_df_stocks = np.log(valid_df_stocks / valid_df_stocks.shift(1))
    log_valid_df_stocks = log_valid_df_stocks.dropna()
    print('天数:', log_valid_df_stocks.shape[0])
    return log_valid_df_stocks


df = pd.DataFrame()
df = ts.get_hist_data('600519', start='2015-01-05', end='2018-12-28')
s600519 = df['close']
s600519.name = '600519'
df = pd.DataFrame()
df = ts.get_hist_data('000651', start='2015-01-05', end='2018-12-28')
s000651 = df['close']
s000651.name = '000651'
df = pd.DataFrame()
df = ts.get_hist_data('000002', start='2015-01-05', end='2018-12-28')
s000002 = df['close']
s000002.name = '000002'
df = pd.DataFrame()
df = ts.get_hist_data('601318', start='2015-01-05', end='2018-12-28')
s601318 = df['close']
s601318.name = '601318'
df = pd.DataFrame()
df = ts.get_hist_data('601857', start='2015-01-05', end='2018-12-28')
s601857 = df['close']
s601857.name = '601857'
print(s601857.head())
data = pd.DataFrame({'600519': s600519, '000651': s000651,
                     '000002': s000002, '601318': s601318, '601857': s601857})

s601857
print(data.head(20))
print(data.shape)
#log_valid_df_stocks = process(data)
#print(log_valid_df_stocks.head())
