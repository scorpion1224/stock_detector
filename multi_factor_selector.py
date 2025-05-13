import os
import tushare as ts
import pandas as pd
import numpy as np
import ta
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
ts.set_token(os.getenv("TUSHARE_TOKEN"))
pro = ts.pro_api()

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 0.02))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 0.04))
ENTRY_PRICE_DISCOUNT = float(os.getenv("ENTRY_PRICE_DISCOUNT", 0.01))


def get_stock_data(ts_code, start_date, end_date):
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df

def calc_factors(df):
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.macd_diff(df['close'])
    df['macd_diff'] = macd
    k = ta.momentum.stoch(df['high'], df['low'], df['close'])
    d = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
    df['kdj_j'] = 3 * k - 2 * d
    boll = ta.volatility.BollingerBands(df['close'], window=20)
    df['boll_lower'] = boll.bollinger_lband()
    return df

def select_signal(df):
    if len(df) < 30:
        return False
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    # MACD金叉
    macd_cross = yesterday['macd_diff'] < 0 and today['macd_diff'] > 0
    # RSI弱势
    rsi_weak = today['rsi'] < 40
    # KDJ超卖
    kdj_oversold = today['kdj_j'] < 20
    # 布林带下轨
    boll_touch = today['close'] <= today['boll_lower']
    return macd_cross and rsi_weak and kdj_oversold and boll_touch

def main():
    st.title("多因子选股系统")
    st.write("基于MACD金叉、RSI弱势、KDJ超卖、布林带下轨的选股系统")
    stock_list = ["000001.SZ", "000333.SZ", "600519.SH"]  # 可批量
    result = []
    for code in stock_list:
        df = get_stock_data(code, "20240101", "20240416")
        if df.empty:
            continue
        df = calc_factors(df)
        if select_signal(df):
            price = df.iloc[-1]['close']
            stop_loss = price * (1 - STOP_LOSS_PERCENT)
            take_profit = price * (1 + TAKE_PROFIT_PERCENT)
            entry = price * (1 - ENTRY_PRICE_DISCOUNT)
            result.append(f"**{code}**\n现价：{price:.2f}\n开仓建议：{entry:.2f}\n止损：{stop_loss:.2f}\n止盈：{take_profit:.2f}\n")
    if result:
        content = "# 多因子选股信号\n" + "\n---\n".join(result)
        st.markdown(content)
    else:
        st.write("无符合条件个股")

if __name__ == "__main__":
    main() 