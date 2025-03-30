This repository demonstrates a Python-based minimum variance investment strategy for US sector ETFs, outperforming the benchmark with an annualized Sharpe ratio of 0.95 over a five-year period.
First, we calculate returns, standard deviation and Sortino ratio of each sector ETF and based on these features we cluster trading days into bullish and bearish.
Given these labels, we train a gradient boosting classifier to predict the label of the next day.
Based on the predicted labels, we construct a minimum variance portfolio out of the bullish sectors. The portfolio is rebalanced daily and outperfroms the benchmark.
In this setting, the benchmark is an equally weighted buy and hold portfolio com prising of all sector ETFs.
The strategy's Sharpe ratio is 0.95 compared to the benchmark's 0.59.
