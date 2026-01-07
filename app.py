import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import requests
from textblob import TextBlob

# Page configuration
st.set_page_config(page_title="Stock Market Visualizer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bullish {
        color: #00cc00;
        font-weight: bold;
    }
    .bearish {
        color: #ff0000;
        font-weight: bold;
    }
    .neutral {
        color: #ffa500;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df, stock
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    # Moving Averages
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    df['EMA_12'] = EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(df['Close'], window=26).ema_indicator()
    
    # RSI
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    
    # Volume Moving Average
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def calculate_fibonacci_levels(df, lookback=100):
    """Calculate Fibonacci retracement levels"""
    recent_data = df.tail(lookback)
    max_price = recent_data['High'].max()
    min_price = recent_data['Low'].min()
    diff = max_price - min_price
    
    levels = {
        '0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '100%': min_price
    }
    return levels

def identify_support_resistance(df, window=20):
    """Identify support and resistance levels"""
    highs = df['High'].rolling(window=window, center=True).max()
    lows = df['Low'].rolling(window=window, center=True).min()
    
    resistance_levels = df[df['High'] == highs]['High'].unique()
    support_levels = df[df['Low'] == lows]['Low'].unique()
    
    # Get top 3 most recent levels
    resistance_levels = sorted(resistance_levels, reverse=True)[:3]
    support_levels = sorted(support_levels)[:3]
    
    return support_levels, resistance_levels

def detect_chart_patterns(df):
    """Detect common chart patterns"""
    patterns = []
    
    # Double Top/Bottom detection (simplified)
    recent_highs = df['High'].tail(50)
    recent_lows = df['Low'].tail(50)
    
    # Head and Shoulders (very simplified detection)
    if len(df) > 60:
        window = df.tail(60)
        peaks = window['High'].nlargest(3)
        if len(peaks) >= 3:
            peaks_list = peaks.tolist()
            if peaks_list[1] > peaks_list[0] and peaks_list[1] > peaks_list[2]:
                patterns.append("Potential Head and Shoulders pattern detected")
    
    return patterns

def analyze_trend(df):
    """Analyze current trend based on moving averages"""
    current_price = df['Close'].iloc[-1]
    sma_20 = df['SMA_20'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    sma_200 = df['SMA_200'].iloc[-1]
    
    if current_price > sma_20 > sma_50 > sma_200:
        return "Strong Uptrend", "bullish"
    elif current_price > sma_20 > sma_50:
        return "Uptrend", "bullish"
    elif current_price < sma_20 < sma_50 < sma_200:
        return "Strong Downtrend", "bearish"
    elif current_price < sma_20 < sma_50:
        return "Downtrend", "bearish"
    else:
        return "Sideways/Consolidation", "neutral"

def get_trading_signals(df):
    """Generate trading signals based on technical indicators"""
    signals = []
    score = 0
    
    current_price = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_signal'].iloc[-1]
    
    # RSI Signals
    if rsi < 30:
        signals.append("🟢 RSI indicates OVERSOLD condition - Potential buying opportunity")
        score += 2
    elif rsi > 70:
        signals.append("🔴 RSI indicates OVERBOUGHT condition - Consider taking profits")
        score -= 2
    else:
        signals.append("🟡 RSI in neutral zone")
    
    # MACD Signals
    if macd > macd_signal and df['MACD'].iloc[-2] <= df['MACD_signal'].iloc[-2]:
        signals.append("🟢 MACD BULLISH crossover - Buy signal")
        score += 2
    elif macd < macd_signal and df['MACD'].iloc[-2] >= df['MACD_signal'].iloc[-2]:
        signals.append("🔴 MACD BEARISH crossover - Sell signal")
        score -= 2
    
    # Moving Average Signals
    sma_20 = df['SMA_20'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    
    if current_price > sma_20 > sma_50:
        signals.append("🟢 Price above both SMA20 and SMA50 - Bullish trend")
        score += 1
    elif current_price < sma_20 < sma_50:
        signals.append("🔴 Price below both SMA20 and SMA50 - Bearish trend")
        score -= 1
    
    # Bollinger Bands
    bb_upper = df['BB_upper'].iloc[-1]
    bb_lower = df['BB_lower'].iloc[-1]
    
    if current_price <= bb_lower:
        signals.append("🟢 Price at lower Bollinger Band - Potential bounce")
        score += 1
    elif current_price >= bb_upper:
        signals.append("🔴 Price at upper Bollinger Band - Potential reversal")
        score -= 1
    
    # Volume Analysis
    current_volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume_MA'].iloc[-1]
    
    if current_volume > avg_volume * 1.5:
        signals.append("🟢 High volume - Strong conviction in price movement")
        score += 0.5
    
    return signals, score

@st.cache_data(ttl=3600)
def fetch_news_sentiment(ticker):
    """Fetch and analyze news sentiment (simplified version)"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return [], 0
        
        sentiments = []
        total_sentiment = 0
        
        for article in news[:5]:  # Analyze top 5 news articles
            title = article.get('title', '')
            # Simple sentiment analysis
            blob = TextBlob(title)
            sentiment = blob.sentiment.polarity
            total_sentiment += sentiment
            
            sentiments.append({
                'title': title,
                'sentiment': sentiment,
                'link': article.get('link', '')
            })
        
        avg_sentiment = total_sentiment / len(sentiments) if sentiments else 0
        return sentiments, avg_sentiment
    except:
        return [], 0

def create_candlestick_chart(df, ticker):
    """Create interactive candlestick chart with indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} Price Chart', 'MACD', 'RSI', 'Volume'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='red', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='red', width=1)), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_diff'], name='MACD Histogram'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=1)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Volume
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=4, col=1)
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">📈 Advanced Stock Market Visualizer & Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", help="e.g., AAPL, GOOGL, MSFT").upper()
        
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        period_label = st.selectbox("Select Time Period", list(period_options.keys()), index=3)
        period = period_options[period_label]
        
        analyze_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Features")
        st.markdown("""
        - Real-time price data
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Moving averages (SMA, EMA)
        - Support & resistance levels
        - Fibonacci retracements
        - News sentiment analysis
        - Trading signals
        """)
    
    if analyze_button:
        with st.spinner(f"Fetching data for {ticker}..."):
            df, stock = fetch_stock_data(ticker, period)
            
            if df is not None and not df.empty:
                # Calculate indicators
                df = calculate_technical_indicators(df)
                
                # Stock Info
                info = stock.info
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    current_price = df['Close'].iloc[-1]
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                    change_pct = (change / df['Close'].iloc[-2]) * 100
                    st.metric("Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
                
                with col3:
                    volume = df['Volume'].iloc[-1]
                    st.metric("Volume", f"{volume:,.0f}")
                
                with col4:
                    high_52w = df['High'].tail(252).max()
                    st.metric("52W High", f"${high_52w:.2f}")
                
                with col5:
                    low_52w = df['Low'].tail(252).min()
                    st.metric("52W Low", f"${low_52w:.2f}")
                
                # Trend Analysis
                st.markdown("---")
                trend, trend_class = analyze_trend(df)
                st.markdown(f"### 📊 Trend Analysis: <span class='{trend_class}'>{trend}</span>", unsafe_allow_html=True)
                
                # Trading Signals
                st.markdown("### 🎯 Trading Signals")
                signals, signal_score = get_trading_signals(df)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    for signal in signals:
                        st.write(signal)
                
                with col2:
                    if signal_score > 3:
                        st.success(f"**Overall Signal: STRONG BUY**\nScore: {signal_score}/10")
                    elif signal_score > 0:
                        st.info(f"**Overall Signal: BUY**\nScore: {signal_score}/10")
                    elif signal_score < -3:
                        st.error(f"**Overall Signal: STRONG SELL**\nScore: {signal_score}/10")
                    elif signal_score < 0:
                        st.warning(f"**Overall Signal: SELL**\nScore: {signal_score}/10")
                    else:
                        st.warning(f"**Overall Signal: NEUTRAL**\nScore: {signal_score}/10")
                
                # Charts
                st.markdown("---")
                st.markdown("### 📈 Interactive Price Chart with Technical Indicators")
                fig = create_candlestick_chart(df, ticker)
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Analysis Details
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📐 Support & Resistance Levels")
                    support, resistance = identify_support_resistance(df)
                    
                    st.write("**Resistance Levels:**")
                    for level in resistance:
                        st.write(f"- ${level:.2f}")
                    
                    st.write("**Support Levels:**")
                    for level in support:
                        st.write(f"- ${level:.2f}")
                    
                    # Fibonacci Levels
                    st.markdown("### 🔢 Fibonacci Retracement Levels")
                    fib_levels = calculate_fibonacci_levels(df)
                    for level_name, level_value in fib_levels.items():
                        st.write(f"- {level_name}: ${level_value:.2f}")
                
                with col2:
                    st.markdown("### 📊 Key Technical Indicators")
                    
                    rsi = df['RSI'].iloc[-1]
                    rsi_color = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
                    st.write(f"{rsi_color} **RSI (14):** {rsi:.2f}")
                    
                    macd = df['MACD'].iloc[-1]
                    macd_signal = df['MACD_signal'].iloc[-1]
                    macd_color = "🟢" if macd > macd_signal else "🔴"
                    st.write(f"{macd_color} **MACD:** {macd:.2f}")
                    st.write(f"**MACD Signal:** {macd_signal:.2f}")
                    
                    st.write(f"**SMA 20:** ${df['SMA_20'].iloc[-1]:.2f}")
                    st.write(f"**SMA 50:** ${df['SMA_50'].iloc[-1]:.2f}")
                    st.write(f"**SMA 200:** ${df['SMA_200'].iloc[-1]:.2f}")
                    
                    bb_upper = df['BB_upper'].iloc[-1]
                    bb_lower = df['BB_lower'].iloc[-1]
                    st.write(f"**Bollinger Upper:** ${bb_upper:.2f}")
                    st.write(f"**Bollinger Lower:** ${bb_lower:.2f}")
                
                # News Sentiment
                st.markdown("---")
                st.markdown("### 📰 News Sentiment Analysis")
                
                with st.spinner("Analyzing recent news..."):
                    news_items, avg_sentiment = fetch_news_sentiment(ticker)
                    
                    if news_items:
                        sentiment_color = "🟢" if avg_sentiment > 0.1 else "🔴" if avg_sentiment < -0.1 else "🟡"
                        st.write(f"{sentiment_color} **Average Sentiment Score:** {avg_sentiment:.3f}")
                        
                        if avg_sentiment > 0.1:
                            st.success("Overall news sentiment is POSITIVE")
                        elif avg_sentiment < -0.1:
                            st.error("Overall news sentiment is NEGATIVE")
                        else:
                            st.info("Overall news sentiment is NEUTRAL")
                        
                        st.markdown("**Recent News Headlines:**")
                        for item in news_items:
                            sentiment_emoji = "🟢" if item['sentiment'] > 0 else "🔴" if item['sentiment'] < 0 else "🟡"
                            st.write(f"{sentiment_emoji} {item['title']} (Sentiment: {item['sentiment']:.3f})")
                    else:
                        st.info("No recent news available for analysis")
                
                # Pattern Detection
                patterns = detect_chart_patterns(df)
                if patterns:
                    st.markdown("---")
                    st.markdown("### 🔍 Detected Chart Patterns")
                    for pattern in patterns:
                        st.write(f"- {pattern}")
                
            else:
                st.error(f"Unable to fetch data for {ticker}. Please check the ticker symbol and try again.")
    
    else:
        st.info("👈 Enter a stock ticker in the sidebar and click 'Analyze Stock' to begin analysis")
        
        st.markdown("---")
        st.markdown("### 🎓 Understanding Technical Analysis")
        
        with st.expander("📊 Key Principles"):
            st.markdown("""
            1. **The market discounts everything**: All known information is already reflected in the stock's price
            2. **Prices move in trends**: Prices tend to move in sustained directions that can be identified
            3. **History tends to repeat itself**: Recurring price patterns result from predictable human psychology
            """)
        
        with st.expander("📈 Technical Indicators Explained"):
            st.markdown("""
            - **RSI (Relative Strength Index)**: Measures momentum. Below 30 = oversold, above 70 = overbought
            - **MACD**: Shows trend direction and momentum. Crossovers generate buy/sell signals
            - **Moving Averages**: Smooth price data to identify trend direction
            - **Bollinger Bands**: Measure volatility. Prices tend to stay within the bands
            - **Volume**: Confirms the strength of price movements
            - **Fibonacci Levels**: Identify potential support/resistance based on key ratios
            """)

if __name__ == "__main__":
    main()
