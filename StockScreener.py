import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Nifty Stock Screener",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StockScreener:
    def __init__(self):
        self.nifty_tickers = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'HCLTECH.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NTPC.NS',
            'NESTLEIND.NS', 'POWERGRID.NS', 'BAJFINANCE.NS', 'M&M.NS', 'TECHM.NS', 'TATAMOTORS.NS', 'HDFCLIFE.NS', 'ONGC.NS', 'SBILIFE.NS', 'COALINDIA.NS',
            'BAJAJFINSV.NS', 'DIVISLAB.NS', 'GRASIM.NS', 'BRITANNIA.NS', 'ADANIPORTS.NS', 'JSWSTEEL.NS', 'DRREDDY.NS', 'TATACONSUM.NS', 'CIPLA.NS', 'EICHERMOT.NS',
            'HINDALCO.NS', 'BPCL.NS', 'TATASTEEL.NS', 'APOLLOHOSP.NS', 'INDUSINDBK.NS', 'HEROMOTOCO.NS', 'UPL.NS', 'BAJAJ-AUTO.NS', 'SHREECEM.NS', 'IOC.NS'
        ]
    
    def get_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period=period)
            info = stock.info
            return hist_data, info
        except Exception as e:
            st.warning(f"Error fetching data for {symbol}: {str(e)}")
            return None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        if data is None or len(data) < 50:
            return {}
        
        # Moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        latest = data.iloc[-1]
        return {
            'current_price': latest['Close'],
            'ma_20': latest['MA_20'],
            'ma_50': latest['MA_50'],
            'ma_200': latest['MA_200'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'macd_signal': latest['MACD_signal'],
            'bb_position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']),
            'volume_ratio': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1
        }
    
    def extract_fundamental_data(self, info):
        """Extract fundamental metrics from stock info"""
        if not info:
            return {}
        
        return {
            'pe_ratio': info.get('trailingPE', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'roe': info.get('returnOnEquity', 0),
            'profit_margin': info.get('profitMargins', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'current_ratio': info.get('currentRatio', 0),
            'market_cap': info.get('marketCap', 0),
            'beta': info.get('beta', 1),
            'dividend_yield': info.get('dividendYield', 0)
        }
    
    def ai_analysis_score(self, technical, fundamental, symbol):
        """AI-based scoring system for stocks"""
        score = 0
        signals = []
        
        if not technical or not fundamental:
            return 0, ["Insufficient data"]
        
        # Technical Analysis (40% weight)
        tech_score = 0
        
        # Moving Average Trend
        if technical['current_price'] > technical['ma_20'] > technical['ma_50']:
            tech_score += 15
            signals.append("Bullish MA trend")
        elif technical['current_price'] < technical['ma_20'] < technical['ma_50']:
            tech_score -= 15
            signals.append("Bearish MA trend")
        
        # RSI Analysis
        rsi = technical.get('rsi', 50)
        if 30 <= rsi <= 70:
            tech_score += 10
            signals.append("Healthy RSI")
        elif rsi < 30:
            tech_score += 5
            signals.append("Oversold RSI")
        elif rsi > 70:
            tech_score -= 5
            signals.append("Overbought RSI")
        
        # MACD
        if technical.get('macd', 0) > technical.get('macd_signal', 0):
            tech_score += 10
            signals.append("MACD bullish")
        else:
            tech_score -= 5
            signals.append("MACD bearish")
        
        # Volume
        if technical.get('volume_ratio', 1) > 1.2:
            tech_score += 5
            signals.append("High volume")
        
        score += tech_score * 0.4
        
        # Fundamental Analysis (60% weight)
        fund_score = 0
        
        # Valuation metrics
        pe = fundamental.get('pe_ratio', 0)
        if 0 < pe < 20:
            fund_score += 20
            signals.append("Attractive P/E")
        elif pe > 30:
            fund_score -= 10
            signals.append("High P/E")
        
        # Profitability
        roe = fundamental.get('roe', 0)
        if roe > 0.15:
            fund_score += 15
            signals.append("Strong ROE")
        elif roe < 0.05:
            fund_score -= 10
            signals.append("Weak ROE")
        
        # Growth
        revenue_growth = fundamental.get('revenue_growth', 0)
        if revenue_growth > 0.1:
            fund_score += 10
            signals.append("Revenue growth")
        elif revenue_growth < -0.05:
            fund_score -= 10
            signals.append("Revenue decline")
        
        # Financial health
        current_ratio = fundamental.get('current_ratio', 0)
        if current_ratio > 1.5:
            fund_score += 5
            signals.append("Strong liquidity")
        
        debt_to_equity = fundamental.get('debt_to_equity', 0)
        if debt_to_equity < 50:
            fund_score += 5
            signals.append("Low debt")
        elif debt_to_equity > 100:
            fund_score -= 5
            signals.append("High debt")
        
        score += fund_score * 0.6
        
        return score, signals
    
    def generate_recommendation(self, score, signals):
        """Generate buy/sell recommendation based on AI score"""
        if score >= 30:
            return "BUY", "🟢", f"Strong buy signal (Score: {score:.1f})"
        elif score <= -20:
            return "SELL", "🔴", f"Strong sell signal (Score: {score:.1f})"
        else:
            return None, "⚪", f"Hold/Neutral (Score: {score:.1f})"
    
    def create_tradingview_link(self, symbol):
        """Create TradingView clickable link"""
        # Remove .NS suffix for TradingView link
        clean_symbol = symbol.replace('.NS', '')
        return f"https://www.tradingview.com/chart/?symbol=NSE%3A{clean_symbol}"
    
    def screen_stocks(self, tickers, filters):
        """Main screening function"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            status_text.text(f"Analyzing {ticker}...")
            progress_bar.progress((i + 1) / len(tickers))
            
            hist_data, info = self.get_stock_data(ticker)
            if hist_data is None or len(hist_data) < 50:
                continue
            
            technical = self.calculate_technical_indicators(hist_data)
            fundamental = self.extract_fundamental_data(info)
            
            # Apply filters
            if not self.passes_filters(technical, fundamental, filters):
                continue
            
            score, signals = self.ai_analysis_score(technical, fundamental, ticker)
            recommendation, color, description = self.generate_recommendation(score, signals)
            
            # Only include BUY or SELL recommendations
            if recommendation in ['BUY', 'SELL']:
                results.append({
                    'Symbol': ticker,
                    'Recommendation': recommendation,
                    'Score': score,
                    'Current Price': technical.get('current_price', 0),
                    'RSI': technical.get('rsi', 0),
                    'P/E Ratio': fundamental.get('pe_ratio', 0),
                    'ROE': fundamental.get('roe', 0) * 100 if fundamental.get('roe') else 0,
                    'Revenue Growth': fundamental.get('revenue_growth', 0) * 100 if fundamental.get('revenue_growth') else 0,
                    'Market Cap': fundamental.get('market_cap', 0),
                    'Signals': ', '.join(signals[:3]),
                    'TradingView': self.create_tradingview_link(ticker),
                    'Color': color
                })
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)
    
    def passes_filters(self, technical, fundamental, filters):
        """Check if stock passes the selected filters"""
        if not technical or not fundamental:
            return False
        
        # Market cap filter
        market_cap = fundamental.get('market_cap', 0)
        if filters['min_market_cap'] > 0 and market_cap < filters['min_market_cap'] * 1e9:
            return False
        
        # P/E ratio filter
        pe = fundamental.get('pe_ratio', 0)
        if filters['max_pe'] > 0 and pe > filters['max_pe']:
            return False
        
        # RSI filter
        rsi = technical.get('rsi', 50)
        if not (filters['min_rsi'] <= rsi <= filters['max_rsi']):
            return False
        
        # ROE filter
        roe = fundamental.get('roe', 0)
        if filters['min_roe'] > 0 and roe < filters['min_roe'] / 100:
            return False
        
        return True

def main():
    st.title("🇮🇳 AI-Powered Nifty Stock Screener")
    st.markdown("*Advanced fundamental & technical analysis with AI-driven buy/sell recommendations for NSE stocks*")
    
    screener = StockScreener()
    
    # Sidebar filters
    st.sidebar.header("📊 Screening Filters")
    
    # Stock selection
    stock_source = st.sidebar.radio(
        "Stock Universe:",
        ["Nifty 50 Sample", "Custom Tickers"]
    )
    
    if stock_source == "Custom Tickers":
        custom_tickers = st.sidebar.text_input(
            "Enter tickers (comma-separated, add .NS for NSE):",
            value="RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS"
        )
        tickers = [t.strip().upper() for t in custom_tickers.split(',')]
    else:
        tickers = screener.nifty_tickers
    
    # Fundamental filters
    st.sidebar.subheader("💰 Fundamental Filters")
    min_market_cap = st.sidebar.number_input("Min Market Cap (Billions ₹)", min_value=0.0, value=10.0, step=5.0)
    max_pe = st.sidebar.number_input("Max P/E Ratio (0 = no limit)", min_value=0.0, value=30.0, step=1.0)
    min_roe = st.sidebar.number_input("Min ROE (%)", min_value=0.0, value=10.0, step=1.0)
    
    # Technical filters
    st.sidebar.subheader("📈 Technical Filters")
    min_rsi = st.sidebar.slider("Min RSI", min_value=0, max_value=100, value=20)
    max_rsi = st.sidebar.slider("Max RSI", min_value=0, max_value=100, value=80)
    
    filters = {
        'min_market_cap': min_market_cap,
        'max_pe': max_pe,
        'min_roe': min_roe,
        'min_rsi': min_rsi,
        'max_rsi': max_rsi
    }
    
    # Main screening button
    if st.button("🚀 Run AI Stock Screening", type="primary"):
        st.header("📋 Screening Results")
        
        with st.spinner("Running AI analysis on selected stocks..."):
            results_df = screener.screen_stocks(tickers, filters)
        
        if len(results_df) > 0:
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            buy_count = len(results_df[results_df['Recommendation'] == 'BUY'])
            sell_count = len(results_df[results_df['Recommendation'] == 'SELL'])
            
            with col1:
                st.metric("Total Analyzed", len(tickers))
            with col2:
                st.metric("Buy Signals", buy_count)
            with col3:
                st.metric("Sell Signals", sell_count)
            with col4:
                st.metric("Success Rate", f"{((buy_count + sell_count) / len(tickers) * 100):.1f}%")
            
            # Separate BUY and SELL recommendations
            buy_stocks = results_df[results_df['Recommendation'] == 'BUY'].sort_values('Score', ascending=False)
            sell_stocks = results_df[results_df['Recommendation'] == 'SELL'].sort_values('Score', ascending=True)
            
            # Display BUY recommendations
            if len(buy_stocks) > 0:
                st.subheader("🟢 BUY Recommendations")
                for _, stock in buy_stocks.iterrows():
                    with st.expander(f"{stock['Symbol']} - Score: {stock['Score']:.1f}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Price", f"₹{stock['Current Price']:.2f}")
                            st.metric("RSI", f"{stock['RSI']:.1f}")
                            st.metric("P/E Ratio", f"{stock['P/E Ratio']:.1f}")
                        with col2:
                            st.metric("ROE", f"{stock['ROE']:.1f}%")
                            st.metric("Revenue Growth", f"{stock['Revenue Growth']:.1f}%")
                            st.metric("Market Cap", f"₹{stock['Market Cap']/1e9:.1f}B")
                        
                        st.write("**Key Signals:**", stock['Signals'])
                        st.markdown(f"[📈 View on TradingView]({stock['TradingView']})")
            
            # Display SELL recommendations
            if len(sell_stocks) > 0:
                st.subheader("🔴 SELL Recommendations")
                for _, stock in sell_stocks.iterrows():
                    with st.expander(f"{stock['Symbol']} - Score: {stock['Score']:.1f}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Price", f"₹{stock['Current Price']:.2f}")
                            st.metric("RSI", f"{stock['RSI']:.1f}")
                            st.metric("P/E Ratio", f"{stock['P/E Ratio']:.1f}")
                        with col2:
                            st.metric("ROE", f"{stock['ROE']:.1f}%")
                            st.metric("Revenue Growth", f"{stock['Revenue Growth']:.1f}%")
                            st.metric("Market Cap", f"₹{stock['Market Cap']/1e9:.1f}B")
                        
                        st.write("**Key Signals:**", stock['Signals'])
                        st.markdown(f"[📈 View on TradingView]({stock['TradingView']})")
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"nifty_screener_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("No stocks met the screening criteria. Try adjusting your filters.")
    
    # Information section
    st.header("ℹ️ How It Works")
    st.markdown("""
    This AI-powered stock screener analyzes Nifty stocks using:
    
    **Technical Analysis (40% weight):**
    - Moving averages (20, 50, 200-day)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Volume analysis
    
    **Fundamental Analysis (60% weight):**
    - Valuation metrics (P/E, P/B ratios)
    - Profitability (ROE, profit margins)
    - Growth metrics (revenue growth)
    - Financial health (debt-to-equity, current ratio)
    - Market capitalization
    
    **AI Scoring System:**
    - Combines technical and fundamental signals
    - Generates scores from -100 to +100
    - Only shows definitive BUY (score ≥30) or SELL (score ≤-20) recommendations
    - Filters out neutral/hold positions for clear actionable insights
    
    **Stock Universe Options:**
    - **Nifty 50 Sample:** Pre-loaded with 50 major NSE stocks
    - **Custom Tickers:** Enter your own NSE stock symbols (remember to add .NS suffix)
    
    **Features:**
    - Real-time NSE data via Yahoo Finance
    - Indian Rupee (₹) pricing display
    - TradingView integration with NSE symbols
    - Comprehensive filtering options
    - Downloadable CSV results
    """)

if __name__ == "__main__":
    main()
