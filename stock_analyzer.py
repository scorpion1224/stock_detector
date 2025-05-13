import os
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class StockAnalyzer:
    def __init__(self):
        try:
            # Try to get credentials from Streamlit secrets first, then environment variables
            tushare_token = st.secrets.get('TUSHARE_TOKEN') or os.getenv('TUSHARE_TOKEN')
            deepseek_api_key = st.secrets.get('DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK_API_KEY')
            
            if not tushare_token:
                st.error("TUSHARE_TOKEN 未设置。请在 Streamlit Cloud 的 Secrets 中配置。")
                st.stop()
                
            if not deepseek_api_key:
                st.error("DEEPSEEK_API_KEY 未设置。请在 Streamlit Cloud 的 Secrets 中配置。")
                st.stop()
            
            # Initialize Tushare
            ts.set_token(tushare_token)
            self.pro = ts.pro_api()
            
            # Initialize Deepseek client
            try:
                self.client = OpenAI(
                    api_key=deepseek_api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                # Test the client
                self.client.models.list()
            except Exception as e:
                st.error(f"OpenAI client initialization failed: {str(e)}")
                st.stop()
            
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")
            st.stop()
        
        # 设置突破检测参数
        self.volatility_threshold = 0.5  # 波动率阈值
        self.min_volume_increase = 1.5   # 最小成交量增幅
        self.rsi_threshold = 40          # RSI阈值

    def get_stock_history(self, stock_code, start_date, end_date):
        """Get historical stock data and calculate technical indicators"""
        # 判断是A股还是港股
        if stock_code.endswith(('.SH', '.SZ')):
            df = self._get_astock_data(stock_code, start_date, end_date)
        elif stock_code.startswith(('00', '02', '03', '06')):  # 港股代码规则
            df = self._get_hkstock_data(stock_code, start_date, end_date)
        else:
            raise ValueError("不支持的股票代码格式")
        
        return self._calculate_indicators(df)

    def _get_astock_data(self, stock_code, start_date, end_date):
        """获取A股数据"""
        df = self.pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        df.reset_index(drop=True, inplace=True)
        return df

    def _get_hkstock_data(self, stock_code, start_date, end_date):
        """获取港股数据"""
        try:
            # 转换港股代码格式（添加.HK后缀）
            hk_code = f"{int(stock_code):05d}.HK"
            
            # 使用tushare获取港股数据
            df = self.pro.hk_daily(ts_code=hk_code, start_date=start_date, end_date=end_date)
            
            if df.empty:
                st.error(f"未找到港股数据: {stock_code}")
                return pd.DataFrame()
            
            # 确保列名一致性
            df = df.rename(columns={
                'vol': 'vol',
                'amount': 'amount',
                'trade_date': 'trade_date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close'
            })
            
            df = df.sort_values('trade_date')
            df.reset_index(drop=True, inplace=True)
            return df
            
        except Exception as e:
            st.error(f"获取港股数据失败: {str(e)}")
            return pd.DataFrame()

    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        if df.empty:
            return df
            
        # Calculate technical indicators
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['boll_mid'] = df['close'].rolling(window=20).mean()
        df['boll_std'] = df['close'].rolling(window=20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
        
        # Volume indicators
        df['vol_ma_5'] = df['vol'].rolling(window=5).mean()
        df['vol_ma_20'] = df['vol'].rolling(window=20).mean()
        
        return df

    def detect_breakout(self, df):
        """Detect potential breakout patterns"""
        if len(df) < 35:
            return False

        recent = df.iloc[-1]
        prev = df.iloc[-2]

        # Check volatility
        vol_std = df['close'][-30:].std()
        if vol_std > self.volatility_threshold:
            return False

        # Check volume increase
        vol_mean_30 = df['vol'][-30:].mean()
        vol_mean_3 = df['vol'][-3:].mean()
        if vol_mean_3 < vol_mean_30 * self.min_volume_increase:
            return False

        # Check MACD
        macd_cross = prev['macd_diff'] < 0 and recent['macd_diff'] > 0

        # Check Bollinger Bands
        boll_break = recent['close'] > recent['boll_upper']

        # Check RSI
        rsi_surge = recent['rsi_14'] > self.rsi_threshold and prev['rsi_14'] < self.rsi_threshold - 10

        return macd_cross or boll_break or rsi_surge

    def plot_stock_data(self, df):
        """Plot stock data with technical indicators"""
        if df.empty:
            return None
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price and indicators
        ax1.plot(df.index, df['close'], label='Close')
        ax1.plot(df.index, df['ma_5'], label='MA5')
        ax1.plot(df.index, df['ma_20'], label='MA20')
        ax1.plot(df.index, df['boll_upper'], 'r--', label='Bollinger Upper')
        ax1.plot(df.index, df['boll_lower'], 'g--', label='Bollinger Lower')
        ax1.set_title('Price and Technical Indicators')
        ax1.legend()
        
        # Volume
        ax2.bar(df.index, df['vol'], label='Volume')
        ax2.plot(df.index, df['vol_ma_5'], 'r-', label='Volume MA5')
        ax2.plot(df.index, df['vol_ma_20'], 'g-', label='Volume MA20')
        ax2.set_title('Volume')
        ax2.legend()
        
        plt.tight_layout()
        return fig

    def analyze_stock_ai(self, df, stock_code):
        """使用AI分析股票数据"""
        try:
            # 准备数据摘要
            recent_data = df.tail(30)
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0] * 100
            vol_change = (recent_data['vol'].iloc[-1] - recent_data['vol'].mean()) / recent_data['vol'].mean() * 100
            
            prompt = f"""
            作为一位专业的股票分析师，请对股票 {stock_code} 的近期走势进行深入分析。以下是关键数据：

            1. 近30日价格变动：{price_change:.2f}%
            2. 近期成交量变化：{vol_change:.2f}%
            3. 最新技术指标：
               - RSI(14): {df['rsi_14'].iloc[-1]:.2f}
               - MACD: {df['macd'].iloc[-1]:.4f}
               - 布林带位置：当前价格 {df['close'].iloc[-1]:.2f}，上轨 {df['boll_upper'].iloc[-1]:.2f}，下轨 {df['boll_lower'].iloc[-1]:.2f}

            请从以下几个方面进行分析：
            1. 技术面分析：
               - 目前的趋势特征
               - 关键支撑和压力位
               - 技术指标信号

            2. 量能分析：
               - 成交量变化特征
               - 是否存在量价配合
               - 主力资金动向判断

            3. 形态研判：
               - 目前处于什么形态
               - 后市发展可能性
               - 需要注意的风险点

            请用专业但易懂的语言进行分析，重点指出关键信号和需要注意的要点。
            """
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"AI 分析生成失败: {str(e)}"

def main():
    st.set_page_config(
        page_title="股票分析系统",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("股票爆发信号分析系统 (支持A股和港股)")
    
    try:
        analyzer = StockAnalyzer()
        
        # Sidebar inputs
        st.sidebar.header("参数设置")
        market = st.sidebar.selectbox("选择市场", ["A股", "港股"])
        
        if market == "A股":
            stock_code_help = "输入股票代码 (例如: 000678.SZ, 600519.SH)"
            default_code = "000678.SZ"
        else:
            stock_code_help = "输入港股代码 (例如: 00700, 02318)"
            default_code = "00700"
            
        stock_code = st.sidebar.text_input("股票代码", default_code, help=stock_code_help)
        start_date = st.sidebar.date_input("开始日期", datetime.now() - timedelta(days=180))
        end_date = st.sidebar.date_input("结束日期", datetime.now())
        
        if st.sidebar.button("分析"):
            with st.spinner('正在获取数据并分析...'):
                # Get and analyze data
                df = analyzer.get_stock_history(
                    stock_code,
                    start_date.strftime('%Y%m%d'),
                    end_date.strftime('%Y%m%d')
                )
                
                if df.empty:
                    st.error("获取数据失败，请检查股票代码是否正确")
                    return
                
                # Display results
                st.subheader("技术分析图表")
                fig = analyzer.plot_stock_data(df)
                if fig:
                    st.pyplot(fig)
                
                # Check for breakout
                if analyzer.detect_breakout(df):
                    st.success("发现潜在突破信号！")
                else:
                    st.info("未发现明显突破信号")
                
                # Display recent data
                st.subheader("最近交易数据")
                st.dataframe(df.tail().style.format({
                    col: '{:.2f}' for col in df.select_dtypes(include=['float64']).columns
                }))
                
                # AI Analysis
                st.subheader("AI 深度分析")
                with st.spinner('正在进行 AI 分析...'):
                    ai_analysis = analyzer.analyze_stock_ai(df, stock_code)
                    st.markdown(ai_analysis)
                    
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 