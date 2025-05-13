# Stock Detector

A powerful stock analysis tool that supports both A-shares and Hong Kong stocks, featuring technical analysis, breakout detection, and AI-powered insights.

## Features

- Real-time stock data analysis for A-shares and Hong Kong stocks
- Technical indicators calculation (MA, RSI, MACD, Bollinger Bands)
- Breakout pattern detection
- AI-powered stock analysis using Deepseek
- Interactive charts and visualizations
- User-friendly Streamlit interface

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock_detector.git
cd stock_detector
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
TUSHARE_TOKEN=your_tushare_token
DEEPSEEK_API_KEY=your_deepseek_api_key
```

5. Run the app:
```bash
streamlit run stock_analyzer.py
```

## Usage

1. Select the market (A-shares or Hong Kong stocks)
2. Enter the stock code
3. Choose the date range for analysis
4. Click "Analyze" to get results

## Requirements

- Python 3.8+
- Tushare API token
- Deepseek API key

## License

MIT License 