# 股票爆发信号分析系统

这是一个基于技术分析和AI的股票爆发信号分析系统，用于识别潜在的股票突破机会。

## 功能特点

- 获取历史股票数据并计算技术指标
- 检测潜在的突破信号
- AI驱动的股票分析
- 可视化技术指标
- 交互式Web界面

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/stock_detector.git
cd stock_detector
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
```bash
cp .env.example .env
```
然后编辑 `.env` 文件，填入你的API密钥和设置。

## 使用方法

1. 运行Streamlit应用：
```bash
streamlit run stock_analyzer.py
```

2. 在Web界面中：
   - 输入股票代码
   - 选择分析的时间范围
   - 点击"分析"按钮查看结果

## 技术指标

系统分析以下技术指标：
- 移动平均线 (MA5, MA20)
- 相对强弱指数 (RSI)
- MACD
- 布林带
- 成交量指标

## 突破信号检测

系统会检测以下突破信号：
- MACD金叉
- 布林带突破
- RSI突破
- 成交量放大

## 注意事项

- 请确保你有有效的Tushare和Deepseek API密钥
- 建议在分析前先进行回测
- 本系统仅供参考，不构成投资建议

## 许可证

MIT License 