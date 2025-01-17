# coding: utf-8
import os
import tushare as ts
import pandas as pd
import numpy as np
from math import isnan
import matplotlib.pyplot as plt
import scipy.optimize as sco
import time

def get_data():
    # 不复权
    stock_info = ts.get_stock_basics()
    code2name = {}
    for code in stock_info.index:
        code2name[code] = stock_info.loc[code, 'name']
    name2code = {v: k for k, v in code2name.items()}
    if os.path.exists('../data/raw_data.xlsx'):
        print('loading from the raw_data.xlsx')
        df_raw = pd.read_excel('../data/raw_data.xlsx', index_col=0)
    else:
        print('getting data...')
        print('about 7-8 minutes...')
        df_sz = ts.get_hist_data('sz', start='2018-01-01',
                                end='2019-03-10')  # p_change就是相对于前一天的收益率
        df_raw = pd.DataFrame()
        df_raw['rt_sz'] = df_sz['close']
        wrong_cnt = 0
        success_cnt = 0
        for code, name in code2name.items():
            try:
                # 抓取不到返回None;get_h_data出问题了。
                df_stock = ts.get_hist_data(
                    code, start='2018-01-01', end='2019-03-10')
                df_tmp2 = pd.DataFrame()
                df_tmp2[code] = df_stock['close']
                df_raw = df_raw.join(df_tmp2)
                del df_tmp2, df_stock
                success_cnt += 1
            except:
                print('skip:', code)
                wrong_cnt += 1
        print('无法获取信息的股票数量:', wrong_cnt)
        print('成功获取信息的股票数量:', success_cnt)
        # 一定要记得逆序，最新的股价在最后一行！方便后续计算收益率
        df_raw = df_raw.reindex(index=df_raw.index[::-1])
        df_raw.to_excel("../data/raw_data.xlsx")
    return df_raw,code2name,name2code



def get_data2():
    ts.set_token('a5c2d8da60c59a16f764a4a45ab338c62682a59e9e14dbe1a2dee6ae')
    pro = ts.pro_api()
    # 查询当前所有正常上市交易的股票列表
    stock_info = pro.stock_basic(
        exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    code2name = {}
    for row in stock_info.iterrows():
        code2name[row[1]['ts_code']] = row[1]['name']
    name2code = {v: k for k, v in code2name.items()}

    if os.path.exists('../data/raw_h_data.xlsx'):
        print('loading from the raw_h_data.xlsx')
        df_raw = pd.read_excel('../data/raw_h_data.xlsx', index_col=0)
    else:
        print('getting data...')
        # print('about 7-8 minutes...') # 使用get_hist_data接口
        print('about 20 minutes or more...')  # 使用pro_bar接口
        df_sz = ts.get_hist_data('sz', start='2018-01-01',
                                end='2019-03-10')  # p_change就是相对于前一天的收益率
        df_raw = pd.DataFrame()
        df_raw['rt_sz'] = df_sz['close']
        # 这个接口和下面的pro_bar不适配，不方便join，因此这个函数有bug。需要在修改DataFrame结构
        wrong_cnt = 0
        success_cnt = 0
        for code, name in code2name.items():
            try:
                # 抓取不到返回None;ts.get_h_data出问题了。
                # 抱歉，您每分钟最多访问该接口200次，权限的具体详情访问：https: // tushare.pro/document/1?doc_id = 108。
                df_stock = ts.pro_bar(ts_code=code, asset='E', adj='qfq',
                           start_date='2018-01-01', end_date='2019-03-10')
                df_tmp2 = pd.DataFrame()
                df_tmp2[code] = df_stock['close']
                df_raw = df_raw.join(df_tmp2)
                del df_tmp2, df_stock
                success_cnt += 1
                time.sleep(0.31)
            except:
                print('skip:', code)
                wrong_cnt += 1
        print('无法获取信息的股票数量:', wrong_cnt)
        print('成功获取信息的股票数量:', success_cnt)
        # 一定要记得逆序，最新的股价在最后一行！方便后续计算收益率
        df_raw = df_raw.reindex(index=df_raw.index[::-1])
        df_raw.to_excel("../data/raw_h_data.xlsx")
    return df_raw,code2name,name2code


def get_data3():
    # 获取前复权的股价数据
    stock_info = ts.get_stock_basics()
    code2name = {}
    for code in stock_info.index:
        code2name[code] = stock_info.loc[code, 'name']
    name2code = {v: k for k, v in code2name.items()}
    if os.path.exists('../data/raw_h_data.xlsx'):
        print('loading from the raw_h_data.xlsx')
        df_raw = pd.read_excel('../data/raw_h_data.xlsx', index_col=0)
    else:
        print('getting data...')
        print('about 7-8 minutes...')
        df_sz = ts.get_hist_data('sz', start='2018-01-01',
                                 end='2019-03-10')  # p_change就是相对于前一天的收益率
        df_raw = pd.DataFrame()
        df_raw['rt_sz'] = df_sz['close']
        df_raw = df_raw.reindex(index=df_raw.index[::-1])
        wrong_cnt = 0
        success_cnt = 0
        for code, name in code2name.items():
            try:
                df_stock = ts.get_k_data(code=code, start='2018-01-01', end='2019-03-10',
                              ktype='D', autype='qfq',
                              index=False,
                              retry_count=3,
                              pause=0.001)
                df_stock = df_stock.set_index("date")
                df_tmp2 = pd.DataFrame()
                df_tmp2[code] = df_stock['close']
                df_raw = df_raw.join(df_tmp2)
                del df_tmp2, df_stock
                success_cnt += 1
            except:
                print('skip:', code)
                wrong_cnt += 1
        print('无法获取信息的股票数量:', wrong_cnt)
        print('成功获取信息的股票数量:', success_cnt)
        # 一定要记得逆序，最新的股价在最后一行！方便后续计算收益率
        df_raw.to_excel("../data/raw_h_data.xlsx")
        print('saved data into the raw_h_data.xlsx')
    return df_raw, code2name, name2code


def get_simple_data():
    # 用于测试
    code2name = {'000651': '格力电器',
                '600519': '贵州茅台',
                '601318': '中国平安',
                '000858': '五粮液',
                '600887': '伊利股份',
                '000333': '美的集团',
                '601166': '兴业银行',
                '600036': '招商银行',
                '601328': '交通银行',
                '600104': '上汽集团'}
    df_raw = pd.DataFrame() 
    for idx,code in enumerate(code2name.keys()):
        df_stock = ts.get_hist_data(
            code, start='2014-05-28', end='2017-05-26')  # 前复权接口有问题
        if idx == 0:
            df_raw[code] = df_stock['close']
        else:
            df_tmp = pd.DataFrame()
            df_tmp[code] = df_stock['close']
            df_raw = df_raw.join(df_tmp)
    name2code = {v:k for k,v in code2name.items()}
    df_raw = df_raw.reindex(index=df_raw.index[::-1])
    return df_raw, code2name, name2code



def get_simple_data2():
    # 用于测试
    code2name = {
                '600519': '贵州茅台',
                '000651': '格力电器',
                '000002': '万科A',
                '601318': '中国平安',
                '601857': '中国石油'
                }
    df_raw = pd.DataFrame() 
    for idx,code in enumerate(code2name.keys()):
        df_stock = ts.get_hist_data(
            code, start='2015-01-05', end='2018-12-28')  # 前复权接口有问题
        if idx == 0:
            df_raw[code] = df_stock['close']
        else:
            df_tmp = pd.DataFrame()
            df_tmp[code] = df_stock['close']
            df_raw = df_raw.join(df_tmp)
    name2code = {v:k for k,v in code2name.items()}
    df_raw = df_raw.reindex(index=df_raw.index[::-1])
    return df_raw, code2name, name2code


def process(df_raw):
    # 处理原始数据，丢弃数据质量较差的股票，并用插值填补缺失值
    code2nan_num = {}
    for code in df_raw.columns:
        code2nan_num[code] = df_raw[code].isnull().sum()
    nan_info = {'0':0,'1':0,'1-10':0,'10-100':0,'100-1000':0,'1000+':0}
    for code,nan_num in code2nan_num.items():
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
    for k,v in nan_info.items():
        print(k,':')
        print(v)
    plt.hist(list(code2nan_num.values()))
    plt.title('nan data distribution')
    valid_codes = []
    invalid_codes = []
    for code,nan_num in code2nan_num.items():
        if nan_num >= 6:
            invalid_codes.append(code)
        else:
            valid_codes.append(code)

    print('删去缺失信息较多的股票，共删去{}只股票'.format(len(invalid_codes)))
    print('筛选后的股票数量:',len(valid_codes))
    
    valid_df_stocks = df_raw.drop(invalid_codes, axis=1, inplace=False)
    valid_df_stocks = valid_df_stocks.interpolate(
        method='linear', limit_direction='forward', axis=0) 
    valid_df_stocks = valid_df_stocks.dropna()
    log_valid_df_stocks = np.log(valid_df_stocks / valid_df_stocks.shift(1))
    log_valid_df_stocks = log_valid_df_stocks.dropna()
    print('天数:', log_valid_df_stocks.shape[0])
    return log_valid_df_stocks


class Solver():
    # 利用单指数模型给股票排名，计算最优投资组合及最优投资比例
    def __init__(self,log_valid_df_stocks, code2name):
        self.log_valid_df_stocks = log_valid_df_stocks
        self.code2name = code2name
        self.code2mean = {}
        self.code2var = {}
        self.code2beta = {}
        self.Rf = 0.0175  # 央行年利率
        self.var_m = np.var(log_valid_df_stocks['rt_sz']) * 252
        self.sorted_stocks = []
        self.code2index = {}
        self.index2code = {}
        self.idx2ratio = {}
        self.selected_stock_n = None
        self.C_opt = None

    def get_selected_stock_n(self):
        return self.selected_stock_n
    def sort_stocks(self,log_valid_df_stocks, code2name):
        # 将股票按单指数模型排序
        stocks = []
        for idx, code in enumerate(log_valid_df_stocks.columns):
            if code == 'rt_sz':
                continue
            self.code2mean[code] = log_valid_df_stocks[code].mean() * 252
            self.code2var[code] = np.var(log_valid_df_stocks[code]) * 252
            self.code2beta[code] = np.cov(log_valid_df_stocks[code], log_valid_df_stocks['rt_sz'])[
                0, 1] * 252 / self.var_m
            stocks.append((code,(self.code2mean[code] - self.Rf)/self.code2beta[code]))
        print('var_m:',self.var_m)
        self.sorted_stocks = sorted(
            stocks, key=lambda stock: stock[1], reverse=True)
        # for idx,stock in enumerate(self.sorted_stocks):
        #     if idx >= 10:
        #         break
        #     print(self.code2name[stock[0]])
        for idx,stock in enumerate(self.sorted_stocks):
            self.code2index[stock[0]] = idx + 1 # index从1开始
            self.index2code[idx+1] = stock[0]
        max_beta = -1
        max_code = None
        for code,beta in self.code2beta.items():
            if beta > max_beta:
                max_beta = beta
                max_code = code
        print('max beta:',max_beta)
        print('the stock with max beta:')
        print(max_code)
        print(self.code2name[max_code])

    def cal_C_opt(self):
        # 计算最优截止率
        idx2sum_a_val = {0:0}
        idx2sum_b_val = {0:0}
        idx2c_val = {}

        def cal_c(index):
            sum_a = idx2sum_a_val[index-1] + (self.code2mean[self.index2code[index]] - self.Rf) * self.code2beta[self.index2code[index]] / self.code2var[self.index2code[index]] 
            sum_b = idx2sum_b_val[index-1] + (self.code2beta[self.index2code[index]] ** 2) / self.code2var[self.index2code[index]]
            c_val = self.var_m * sum_a / (1 + self.var_m * sum_b)
            idx2sum_a_val[index] = sum_a
            idx2sum_b_val[index] = sum_b
            idx2c_val[index] = c_val
            assert not isnan(c_val),'nan with {}'.format(index)

        for i in range(1,len(self.code2index.keys())+1):
            cal_c(i)

        # select
        for i in range(len(self.sorted_stocks)):
            if self.sorted_stocks[i][1] <= idx2c_val[i+1]:
                break
        selected_stock_n = i
        print(selected_stock_n) #最后一个符合条件的股票index，index从1开始。
        self.selected_stock_n = selected_stock_n
        C_opt = idx2c_val[selected_stock_n]
        self.C_opt = C_opt
        print('C_opt:',C_opt)

    def cal_ratio(self):
        # 计算最优投资比例
        idx2z_val = {}
        z_val_sum = 0
        for i in range(1,self.selected_stock_n+1):
            beta = self.code2beta[self.index2code[i]]
            z_val = beta /self.code2var[self.index2code[i]] * (((self.code2mean[self.index2code[i]] - self.Rf) / beta) - self.C_opt)
            idx2z_val[i] = z_val
            z_val_sum += z_val
        for i in range(1,self.selected_stock_n+1):
            self.idx2ratio[i] = idx2z_val[i] / z_val_sum


    def show_top_k(self,k):
        # 显示最优投资组合及投资比例
        stocks_ratio = []
        opt_weights = []
        for idx,ratio in self.idx2ratio.items():
            stocks_ratio.append((idx,ratio))
            # print(idx)
        for code,index in self.code2index.items():
            if index > len(self.idx2ratio.keys()):
                break
            # print(code) # for debug
            # print(index)
            opt_weights.append(self.idx2ratio[index])
        sorted_stock_ratio = sorted(stocks_ratio,key = lambda stock : stock[1],reverse = True)
        print('按最优投资组合公式计算,各股票投资比例:')
        for i in range(k):
            stock = sorted_stock_ratio[i]
            print(self.index2code[stock[0]],'\t',self.code2name[self.index2code[stock[0]]],'\t',stock[1])
        return opt_weights

    def select_stocks(self,simple_mode = False):
        # 返回最优投资组合的股价信息
        selected_stocks = pd.DataFrame()
        for code, index in self.code2index.items():
            # print('selecting...')
            # print(index)
            if (simple_mode and index > 10) or index > self.selected_stock_n:
                break
            selected_stocks[code] = self.log_valid_df_stocks[code]
        selected_stocks_dropna = selected_stocks.dropna()
        return selected_stocks_dropna


def plot_frontier(selected_stocks, code2name, opt1):
    # 画出有效边界，并通过最优化方法计算最优投资比例
    selected_stock_n = selected_stocks.shape[1]
    print('selected_stock_n:', selected_stock_n)
    np.random.seed(123)
    num_portfolios = 300000
    all_weights = np.zeros((num_portfolios, selected_stock_n))
    returns = np.zeros(num_portfolios) # Expected return
    vols = np.zeros(num_portfolios) # Expected volatility
    sharpes = np.zeros(num_portfolios)

    stocks_mean = selected_stocks.mean() * 252  # 以一年252个交易日计算
    stocks_cov = selected_stocks.cov() * 252

    def get_ret_vol_sr(weights):
        weights = np.array(weights)
        ret = np.sum(stocks_mean * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(stocks_cov, weights)))
        sr = ret / vol
        return np.array([ret, vol, sr])
        
    max_sr_ret, max_sr_vol, max_sr = get_ret_vol_sr(opt1)
    print('sharpe ratio of the formula-opt portfolio: {}'.format(max_sr))
    print('return of the formula-opt portfolio: {}'.format(max_sr_ret))
    print('volatility of the formula-opt portfolio: {}'.format(max_sr_vol))

    for i in range(num_portfolios):
        # Weights
        weights = np.array(np.random.random(selected_stock_n))
        weights = weights/np.sum(weights)
        # Save weights
        all_weights[i,:] = weights

        returns[i],vols[i],sharpes[i] = get_ret_vol_sr(weights)
    
    max_sr_ret = returns[sharpes.argmax()]
    max_sr_vol = vols[sharpes.argmax()]
    print('max sharpe ratio in simulation: {}'.format(sharpes.max()))
    print('return with max sharpe ratio in simulation: {}'.format(max_sr_ret))
    print('volatility with max sharpe ratio in simulation: {}'.format(max_sr_vol))
    # print('index: {}'.format(sharpes.argmax()))

    print('达到最高夏普率条件下(模拟),各股票投资比例:')
    opt_weights = all_weights[sharpes.argmax()]
    for idx,col in enumerate(selected_stocks.columns):
        print(code2name[col], '\t', opt_weights[idx])
        
    plt.figure(1)
    plt.title('模拟结果')
    plt.figure(figsize=(12,8))
    plt.scatter(vols, returns, c=sharpes,cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # red dot
    plt.show()

    def neg_sharpe(weights):
    # the number 2 is the sharpe ratio index from the get_ret_vol_sr
        return -1 * get_ret_vol_sr(weights)[2] 

    cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights)-1})
    bounds = tuple((0, 1) for _ in range(selected_stock_n))
    init_guess = selected_stock_n * [1. / selected_stock_n,]
    opts = sco.minimize(neg_sharpe, init_guess, method='SLSQP',  bounds=bounds, constraints=cons)
    max_sr_ret,max_sr_vol,max_sharpe = get_ret_vol_sr(opts['x']).round(3)
    print('max sharpe ratio in optimization: {}'.format(max_sharpe))
    print('return with max sharpe ratio in optimization: {}'.format(max_sr_ret))
    print('volatility with max sharpe ratio in optimization: {}'.format(max_sr_vol))
    opt_weights = opts['x']
    print('达到最高夏普率条件下(最优化),各股票投资比例:')
    for idx, col in enumerate(selected_stocks.columns):
        print(code2name[col], '\t', opt_weights[idx])

    # frontier_y = np.linspace(0.4,0.8,200)
    # frontier_y = np.linspace(0.05, 0.3, 200)
    # frontier_y = np.linspace(0.425,0.925,200)
    frontier_y = np.linspace(0.425,0.9,200)
    def minimize_vol(weights):
        return get_ret_vol_sr(weights)[1]
    frontier_x = []
    for possible_return in frontier_y:
        cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights)-1},
                {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
        
        result = sco.minimize(minimize_vol,init_guess,method='SLSQP', bounds=bounds, constraints=cons)
        frontier_x.append(result['fun'])


    # 修正曲线，要取上半段
    start_idx = np.array(frontier_x).argmin()
    end_idx = frontier_y.argmax()
    # print(start_idx)
    # print(end_idx)
    # frontier_x = frontier_x[start_idx:end_idx+1]
    # frontier_y = frontier_y[start_idx:end_idx+1]
    # print(frontier_x)
    # print(frontier_y)
    plt.figure(figsize=(12,8))
    plt.scatter(vols, returns, c=sharpes, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.plot(frontier_x,frontier_y, 'r--', linewidth=3)
    plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50)  # red dot
    plt.title('Efficient Frontier')
    plt.savefig('frontier.png')
    plt.show()
    return opt_weights


def max_drawdown(prices):
    # 计算最大回撤率
    # print(prices)
    i = np.argmax((np.maximum.accumulate(prices) - prices) /
                  np.maximum.accumulate(prices))
    if i == 0:
        return 0
    j = np.argmax(prices[:i])
    return (prices[j] - prices[i]) / (prices[j])



def backtest(selected_stocks,opt1, opt2):
    # 进行回测
    selected_stocks_history = selected_stocks
    if os.path.exists('../data/raw_backtest_data.xlsx'):
        print('loading from the raw_backtest_data.xlsx')
        df_raw = pd.read_excel('../data/raw_backtest_data.xlsx', index_col=0)
    else:
        print('getting data...')
        df_sz = ts.get_hist_data('sz', start='2019-03-11',
                                 end='2019-03-31')  
        df_raw = pd.DataFrame()
        df_raw['rt_sz'] = df_sz['close']
        df_raw = df_raw.reindex(index=df_raw.index[::-1])
        wrong_cnt = 0
        success_cnt = 0
        for code in selected_stocks_history.columns:
            try:
                df_stock = ts.get_k_data(code=code, start='2019-03-11', end='2019-03-31',
                              ktype='D', autype='qfq',
                              index=False,
                              retry_count=3,
                              pause=0.001)
                df_stock = df_stock.set_index("date")
                df_tmp2 = pd.DataFrame()
                df_tmp2[code] = df_stock['close']
                df_raw = df_raw.join(df_tmp2)
                del df_tmp2, df_stock
                success_cnt += 1
            except:
                print('skip:', code)
                wrong_cnt += 1
        print('无法获取信息的股票数量:', wrong_cnt)
        print('成功获取信息的股票数量:', success_cnt)
        
        df_raw.to_excel("../data/raw_backtest_data.xlsx")
        print('saved data into the raw_backtest_data.xlsx')
    df_stocks = df_raw.interpolate(
            method='linear', limit_direction='forward', axis=0)
    sz_raw = df_stocks.iloc[0,0]
    opt1 = np.array(opt1)
    bought_price1 = np.inner(opt1, df_stocks.iloc[0,1:].values)
    bought_price2 = np.inner(opt2, df_stocks.iloc[0,1:].values)
    day_n = df_stocks.shape[0]
    sz_return_list = [] 
    return1_list = [] 
    return2_list = []
    prices1 = []
    prices1.append(bought_price1)
    prices2 = []
    prices2.append(bought_price2)
    for idx in range(1,day_n):
        sz_return_list.append(df_stocks.iloc[idx,0] / sz_raw -1)
        price1 = np.inner(opt1, df_stocks.iloc[idx, 1:].values)
        price2 = np.inner(opt2, df_stocks.iloc[idx, 1:].values)
        prices1.append(price1)
        prices2.append(price2)
        return1_list.append(price1 / bought_price1 -1)
        return2_list.append(price2 / bought_price2 -1)
    print('formula最大回测率', max_drawdown(prices1))
    print('opt最大回测率', max_drawdown(prices2))
    dates = df_stocks.index.values
    dates = dates[1:]
    plt.figure(1)
    plt.title("Back Test")
    plt.plot(dates,sz_return_list,'green',label = 'sz')
    plt.plot(dates, return1_list, 'red', label='formula')
    plt.plot(dates, return2_list, 'blue', label='opt')
    plt.legend()  # 显示图例
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.savefig('back_test.png')
    plt.show()

    # 计算实时收益率
    print('getting data...')
    df_sz = ts.get_hist_data('sz', start='2019-06-01',
                                end='2019-06-25')  
    df_raw = pd.DataFrame()
    df_raw['rt_sz'] = df_sz['close']
    df_raw = df_raw.reindex(index=df_raw.index[::-1])
    wrong_cnt = 0
    success_cnt = 0
    for code in selected_stocks_history.columns:
        try:
            df_stock = ts.get_k_data(code=code, start='2019-06-01',
                                     end='2019-06-25',
                            ktype='D', autype='qfq',
                            index=False,
                            retry_count=3,
                            pause=0.001)
            df_stock = df_stock.set_index("date")
            df_tmp2 = pd.DataFrame()
            df_tmp2[code] = df_stock['close']
            df_raw = df_raw.join(df_tmp2)
            del df_tmp2, df_stock
            success_cnt += 1
        except:
            print('skip:', code)
            wrong_cnt += 1
    print('无法获取信息的股票数量:', wrong_cnt)
    print('成功获取信息的股票数量:', success_cnt)
    df_stocks = df_raw.interpolate(
        method='linear', limit_direction='forward', axis=0)
    bought_price = np.inner(opt2, df_stocks.iloc[0, 1:].values)
    cur_price = np.inner(opt2, df_stocks.iloc[-1, 1:].values)
    print('return in reality:',cur_price / bought_price - 1)




if __name__ == '__main__':
    df_raw, code2name, name2code = get_data3() # 获取数据
    log_valid_df_stocks = process(df_raw) # 处理数据，丢弃不符条件的股票数据，并用线性插值填补缺失值
    solver = Solver(log_valid_df_stocks,code2name) # 初始化求解器
    solver.sort_stocks(log_valid_df_stocks, code2name) # 对股票排序
    solver.cal_C_opt() # 计算最优截止率
    solver.cal_ratio() # 计算最优投资比例
    selected_stock_n = solver.get_selected_stock_n() # 返回最优投资组合包含的股票数量
    opt1 = solver.show_top_k(selected_stock_n) # 显示最优投资组合包含的股票及其比例
    selected_stocks = solver.select_stocks()  # 返回最优投资组合包含的股票的股价数据
    opt2 = plot_frontier(selected_stocks, code2name, opt1) # 画出有效边界，并计算最高夏普率条件下的最优投资比例，并计算各项评价投资组合的指标
    backtest(selected_stocks,opt1,opt2) # 进行回测


    
