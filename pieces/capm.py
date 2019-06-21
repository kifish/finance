#环境&数据准备
import sys as sy
import numpy as np
import pandas as pd
import tushare as ts
import pyecharts as pye
from sklearn import datasets as ds
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pyecharts as pye

#读入美的“000333”2017-01-01 到 2018-11-08复权后数据
#美的取了前复权数据
df_000333 = ts.get_hist_data('600446', start='2018-01-01', end='2019-06-10')
# 000651 格力
# stocks = ['000333','000651']
#深成指数木有复权一说
df_sz = ts.get_hist_data('sz', start='2018-01-01', end='2019-06-10')

#数据准备
df_tmp1 = pd.DataFrame()
df_tmp2 = pd.DataFrame()
df_tmp = pd.DataFrame()
#df_tmp1['rt_000333'] =(df_000333['close'].diff(1))/df_000333['close'].shift(1)
#df_tmp2['rt_sz'] =(df_sz['close'].diff(1))/df_sz['close'].shift(1)
#df_tmp = df_tmp1.join(df_tmp2)

print(df_000333.head())

df_tmp['rt_000333'] = df_000333['p_change']
df_tmp['rt_sz'] = df_sz['p_change']
print(df_tmp.head())

df_tmp = df_tmp.dropna()  # 丢弃有nan的行。
print(df_tmp.shape[0])
print(df_tmp.head())
#del df_tmp1
#del df_tmp2


#计算Beta系数

cov_sm = np.cov(df_tmp.rt_000333, df_tmp.rt_sz)[0, 1]
var_m = np.var(df_tmp.rt_sz)
Beta = cov_sm/var_m  # 单个股票的beta是股票与市场的协方差除以市场利润方差

print(cov_sm)
print(np.cov(df_tmp['rt_000333'], df_tmp['rt_sz'])[0, 1])
print(var_m)
print(Beta)

#下面的例子是为了告诉大家Beta可通过线性回归系数来求得， 效果有一点不一样一样的。
#from scipy import stats
#Beta_ln = stats.linregress(df_tmp.rt_sz,  df_tmp.rt_000333)
#这个算法不是很好， 不应该把SZ的日均收益率转换成年化收益率的，用365天来差分，可能合理一些
Erm = df_tmp.rt_sz.mean()*365  # 计算年化市场期望收益率,这里直接乘以365是有问题的。
Rf = 0.015  # 央行一年期定存利率， 也可也换成10年期债券， 其实每个人眼里的无风险收益的欧是不一样的。
attribute = (df_tmp.rt_000333.mean() - Rf) / Beta

#Ci =
#根据公司Ers = rf + Beta*(Erm - rf)
Ers = Rf + Beta*(Erm - Rf)
print('attr:', attribute)
print('Bata = ' + str(Beta))
print('Rf = ' + str(Rf))
print('Erm = ' + str(Erm))
print('Ers = ' + str(Ers))
