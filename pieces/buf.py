import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import scipy.optimize as sco

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

data = pd.DataFrame({'600519': s600519, '000651': s000651,
                     '000002': s000002, '601318': s601318, '601857': s601857})
data = data.dropna()

print(data.head())

print(data.shift(1).head())
data = np.log(data / data.shift(1))

print(data.head(20))
print(data.shape)
print(data.index)
# returns_annual = data.mean() * 252
# cov_annual = data.cov() * 252

# number_assets = 5
# weights = np.random.random(number_assets)
# weights /= np.sum(weights)

# portfolio_returns = []
# portfolio_volatilities = []
# sharpe_ratio = []
# for single_portfolio in range(50000):
#       weights = np.random.random(number_assets)
#       weights /= np.sum(weights)
#       returns = np.dot(weights, returns_annual)
#       volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
#       portfolio_returns.append(returns)
#       portfolio_volatilities.append(volatility)
#       sharpe = returns / volatility
#       sharpe_ratio.append(sharpe)

# portfolio_returns = np.array(portfolio_returns)
# portfolio_volatilities = np.array(portfolio_volatilities)

# plt.style.use('seaborn-dark')
# plt.figure(figsize=(9, 5))
# plt.scatter(portfolio_volatilities, portfolio_returns,
#             c=sharpe_ratio, cmap='RdYlGn', edgecolors='black', marker='o')
# plt.grid(True)
# plt.xlabel('expected volatility')
# plt.ylabel('expected return')
# plt.colorbar(label='Sharpe ratio')
# plt.show()

# def statistics(weights):
#     weights = np.array(weights)
#     pret = np.sum(data.mean() * weights) * 252
#     pvol = np.sqrt(np.dot(weights.T, np.dot(data.cov() * 252, weights)))
#     return np.array([pret, pvol, pret / pvol])


# def min_func_sharpe(weights):
#     return -statistics(weights)[2]


# def plot_frontier(selected_stocks):
#     selected_stock_n = selected_stocks.shape[1]
#     print('selected_stock_n:', selected_stock_n)
#     np.random.seed(123)
#     num_portfolios = 6000
#     all_weights = np.zeros((num_portfolios, selected_stock_n))
#     returns = np.zeros(num_portfolios)  # Expected return
#     vols = np.zeros(num_portfolios)  # Expected volatility
#     sharpes = np.zeros(num_portfolios)

#     stocks_mean = selected_stocks.mean() * 252  # 以一年252个交易日计算
#     stocks_cov = selected_stocks.cov() * 252

#     def get_ret_vol_sr(weights):
#         weights = np.array(weights)
#         ret = np.sum(stocks_mean * weights)
#         vol = np.sqrt(np.dot(weights.T, np.dot(stocks_cov, weights)))
#         sr = ret / vol
#         return np.array([ret, vol, sr])

#     for i in range(num_portfolios):
#         # Weights
#         weights = np.array(np.random.random(selected_stock_n))
#         weights = weights/np.sum(weights)
#         # Save weights
#         all_weights[i, :] = weights

#         returns[i], vols[i], sharpes[i] = get_ret_vol_sr(weights)

#     max_sr_ret = returns[sharpes.argmax()]
#     max_sr_vol = vols[sharpes.argmax()]
#     print('max sharpe ratio in simulation: {}'.format(sharpes.max()))
#     print('return with max sharpe ratio in simulation: {}'.format(max_sr_ret))
#     print('volatility with max sharpe ratio in simulation: {}'.format(max_sr_vol))
#     # print('index: {}'.format(sharpes.argmax()))

#     print('达到最高夏普率条件下,各股票投资比例:')
#     print(all_weights[sharpes.argmax()])
#     plt.figure(1)
#     plt.title('模拟结果')
#     plt.figure(figsize=(12, 8))
#     plt.scatter(vols, returns, c=sharpes, cmap='viridis')
#     plt.colorbar(label='Sharpe Ratio')
#     plt.xlabel('Volatility')
#     plt.ylabel('Return')
#     plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50)  # red dot
#     plt.show()

#     def neg_sharpe(weights):
#           # the number 2 is the sharpe ratio index from the get_ret_vol_sr
#         return -1 * get_ret_vol_sr(weights)[2]

#     cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights)-1})
#     bounds = tuple((0, 1) for _ in range(selected_stock_n))
#     init_guess = selected_stock_n * [1. / selected_stock_n, ]
#     opts = sco.minimize(neg_sharpe, init_guess, method='SLSQP',
#                         bounds=bounds, constraints=cons)
#     max_sr_ret, max_sr_vol, max_sharpe = get_ret_vol_sr(opts['x']).round(3)
#     print('max sharpe ratio in optimization: {}'.format(max_sharpe))
#     print('return with max sharpe ratio in optimization: {}'.format(max_sr_ret))
#     print('volatility with max sharpe ratio in optimization: {}'.format(max_sr_vol))

#     frontier_y = np.linspace(0.05, 0.3, 200)

#     def minimize_vol(weights):
#         return get_ret_vol_sr(weights)[1]
#     frontier_x = []
#     for possible_return in frontier_y:
#         cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights)-1},
#                 {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})

#         result = sco.minimize(minimize_vol, init_guess,
#                               method='SLSQP', bounds=bounds, constraints=cons)
#         frontier_x.append(result['fun'])

#     # 修正曲线，要取上半段
#     start_idx = np.array(frontier_x).argmin()
#     end_idx = frontier_y.argmax()
#     print(start_idx)
#     print(end_idx)
#     # frontier_x = frontier_x[start_idx:end_idx+1]
#     # frontier_y = frontier_y[start_idx:end_idx+1]
#     print(frontier_x)
#     print(frontier_y)
#     plt.figure(figsize=(12, 8))
#     plt.scatter(vols, returns, c=sharpes, cmap='viridis')
#     plt.colorbar(label='Sharpe Ratio')
#     plt.xlabel('Volatility')
#     plt.ylabel('Return')
#     plt.plot(frontier_x, frontier_y, 'r--', linewidth=3)
#     plt.title('efficient frontier')
#     plt.savefig('frontier.png')
#     plt.show()

# bnds = tuple((0, 1) for x in range(number_assets))
# cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# opts = sco.minimize(min_func_sharpe, number_assets *
#                     [1. / number_assets, ], method='SLSQP',  bounds=bnds, constraints=cons)
# opts['x'].round(3)  # 得到各股票权重
# statistics(opts['x']).round(3)  # 得到投资组合预期收益率、预期波动率以及夏普比率

# plot_frontier(data)
