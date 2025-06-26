# Solana Trading Strategy: Statistical and Machine Learning Approach
## Overview
This project demonstrates a full data science and quant workflow applied to the Solana (SOL) cryptocurrency. It includes data acquisition, statistical analysis, probability modeling, feature engineering, and the development of rule-based and machine learning-based trading strategies.

The goal is to explore the dynamics of daily SOL price movements and to build data-driven strategies that can potentially outperform naive benchmarks.

# Project Steps
*Data Collection*

Retrieved 1-day OHLCV data for Solana (SOL/USDT) over the past 365 days using the Binance API.

Data Cleaning & Preprocessing

Removed missing values and handled irregularities.

Calculated daily returns and log-returns.

Exploratory Data Analysis (EDA)

Analyzed distribution of returns and autocorrelation.

Examined correlations between price and volume-based features.

Probability Distribution Modeling

Fitted different probability distributions (Normal, Exponential, Gamma, Weibull, Poisson) to daily return data.

Evaluated goodness-of-fit and tail behavior to identify suitable risk modeling approaches.

Rule-Based Strategy

Developed a simple volatility-based strategy using technical indicators (e.g., ATR, RSI, SMA).

Simulated performance over the historical data.

Machine Learning Model

Engineered features including volatility regimes, momentum indicators, and market stress signals.

Trained an XGBoost classifier to predict next-day return direction.

Performed feature selection using SHAP values.

Evaluated model performance and applied it in a backtesting framework.

Tools & Libraries
Python (pandas, numpy, scikit-learn, xgboost, scipy, statsmodels, shap, matplotlib, seaborn)

Binance API

Custom backtesting logic

Key Insights
Volatility was a significant driver of return dynamics in the considered period.

Some non-Gaussian distributions (e.g., Gamma, Weibull) fit the empirical returns better than the Normal distribution.

The XGBoost model showed predictive value with selected features, especially during regime changes and volume spikes.

Next Steps
Extend the feature set with macro or on-chain indicators.

Use time series models (e.g., ARIMA, GARCH) for volatility forecasting.

Explore ensemble models or deep learning for signal generation.

About
This repository is part of my data science & quant portfolio. My background combines financial knowledge with statistical modeling and Python-based machine learning. I am actively looking for opportunities in quantitative research, algorithmic trading, and data-driven strategy development.

