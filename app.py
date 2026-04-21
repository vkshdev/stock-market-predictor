"""
Stock Market Predictor with XGBoost
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data_handler import DataHandler
from indicators import TechnicalIndicators
from sentiment import SentimentAnalyzer
from predictor import StockPredictor

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    h1 { color: #1f77b4; text-align: center; }
</style>
""", unsafe_allow_html=True)

if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler()
    st.session_state.sentiment_analyzer = SentimentAnalyzer()

st.markdown("""
<div style="text-align: center; padding: 1.5rem; background: linear-gradient(90deg, #1f77b4, #2ca02c); color: white; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0;">Stock Market Predictor</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
         Prediction of NIFTY 50, SENSEX & Individual Indian Stocks
    </p>
</div>
""", unsafe_allow_html=True)

def create_prediction_chart(prediction_data):
    data = prediction_data['data']
    predictions = prediction_data['predictions']
    recent_data = data.tail(60)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Price & Predictions', 'Volume'],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Scatter(x=recent_data.index, y=recent_data['Close'],
                  mode='lines', name='Historical',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    pred_dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in predictions]
    pred_prices = [p['predicted_price'] for p in predictions]
    connection_dates = [recent_data.index[-1]] + pred_dates
    connection_prices = [recent_data['Close'].iloc[-1]] + pred_prices
    
    fig.add_trace(
        go.Scatter(x=connection_dates, y=connection_prices,
                  mode='lines+markers', name='Predictions',
                  line=dict(color='red', width=2, dash='dash'),
                  marker=dict(size=8, color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=recent_data.index, y=recent_data['Volume'],
               name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    fig.update_yaxes(title_text="Price (Rs)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def show_technical_analysis(prediction_data, predictor):
    data = prediction_data['data']
    latest = data.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Moving Averages:**")
        st.write(f"Price vs SMA20: {((latest['Close'] / latest['SMA_20'] - 1) * 100):+.1f}%")
        st.write(f"Price vs SMA50: {((latest['Close'] / latest['SMA_50'] - 1) * 100):+.1f}%")
        st.markdown("**Oscillators:**")
        st.write(f"RSI: {latest['RSI']:.1f}")
        st.write(f"Stochastic K: {latest['Stoch_K']:.1f}")
    
    with col2:
        st.markdown("**Momentum:**")
        st.write(f"MACD: {latest['MACD']:.2f}")
        st.write(f"5-day: {latest['Momentum_5']*100:+.1f}%")
        st.markdown("**Volatility:**")
        st.write(f"Annual: {latest['Volatility']*100:.1f}%")
        st.write(f"BB Position: {latest['BB_Position']:.2f}")
    
    if predictor:
        importance_df = predictor.get_feature_importance(top_n=10)
        if importance_df is not None:
            st.markdown("**Top 10 Important Features:**")
            st.dataframe(importance_df, use_container_width=True)

def show_sentiment_analysis(prediction_data):
    sentiment = prediction_data['sentiment']
    st.markdown(f"**Overall Sentiment:** {sentiment['sentiment_trend']}")
    st.markdown(f"**Sentiment Score:** {sentiment['avg_sentiment']:.2f}")
    st.markdown(f"**Confidence:** {sentiment['confidence']*100:.1f}%")
    st.markdown(f"**Articles Analyzed:** {sentiment['article_count']}")
    
    if sentiment['avg_sentiment'] > 0.2:
        st.success("Positive sentiment may support price growth")
    elif sentiment['avg_sentiment'] < -0.2:
        st.error("Negative sentiment may pressure prices")
    else:
        st.info("Neutral sentiment - technicals drive price")

def show_recommendations(prediction_data):
    day1_pred = prediction_data['predictions'][0]
    change_pct = day1_pred['change_pct']
    confidence = day1_pred['confidence']
    
    if change_pct > 2 and confidence > 70:
        recommendation = "STRONG BUY"
        reasoning = "High confidence bullish prediction"
    elif change_pct > 1 and confidence > 60:
        recommendation = "BUY"
        reasoning = "Moderate bullish prediction"
    elif change_pct < -2 and confidence > 70:
        recommendation = "STRONG SELL"
        reasoning = "High confidence bearish prediction"
    elif change_pct < -1 and confidence > 60:
        recommendation = "SELL"
        reasoning = "Moderate bearish prediction"
    else:
        recommendation = "HOLD"
        reasoning = "Neutral or low confidence"
    
    st.markdown(f"### {recommendation}")
    st.markdown(f"**Reasoning:** {reasoning}")
    st.markdown(f"**Confidence:** {confidence}%")
    
    current_price = prediction_data['current_price']
    predicted_price = day1_pred['predicted_price']
    
    st.markdown("**Suggested Levels:**")
    st.write(f"Entry: Rs {current_price * 0.998:.2f}")
    st.write(f"Target: Rs {predicted_price * 1.01:.2f}")
    st.write(f"Stop Loss: Rs {current_price * 0.97:.2f}")

def display_prediction_results(prediction_data, predictor):
    st.success("Analysis Complete")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"Rs {prediction_data['current_price']:,.2f}")
    
    with col2:
        day1_pred = prediction_data['predictions'][0]
        st.metric("Tomorrow's Prediction", f"Rs {day1_pred['predicted_price']:,.2f}",
                 f"{day1_pred['change']:+,.2f} ({day1_pred['change_pct']:+.2f}%)")
    
    with col3:
        sentiment = prediction_data['sentiment']
        st.metric("Sentiment", sentiment['sentiment_trend'], f"{sentiment['avg_sentiment']:.2f}")
    
    with col4:
        dir_acc = prediction_data['cv_results']['avg_directional_accuracy']
        st.metric("Directional Accuracy", f"{dir_acc:.1f}%")
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R2 Score", f"{prediction_data['cv_results']['avg_test_r2']:.3f}")
    with col2:
        st.metric("MAPE", f"{prediction_data['cv_results']['avg_test_mape']:.2f}%")
    with col3:
        st.metric("MAE", f"Rs {prediction_data['cv_results']['avg_test_mae']:.2f}")
    with col4:
        st.metric("Algorithm", "XGBoost")

    st.subheader("Multi-Day Predictions")
    pred_df = pd.DataFrame(prediction_data['predictions'])
    pred_df_display = pred_df[['date', 'predicted_price', 'change', 'change_pct', 'confidence', 'direction']].copy()
    pred_df_display.columns = ['Date', 'Predicted Price', 'Change', 'Change %', 'Confidence %', 'Direction']
    st.dataframe(pred_df_display, use_container_width=True)
    
    create_prediction_chart(prediction_data)
    
    tab1, tab2, tab3 = st.tabs(["Technical Analysis", "Sentiment", "Recommendations"])
    with tab1:
        show_technical_analysis(prediction_data, predictor)
    with tab2:
        show_sentiment_analysis(prediction_data)
    with tab3:
        show_recommendations(prediction_data)

def display_portfolio_results(results):
    st.success(f"Analyzed {len(results)} stocks successfully")
    
    summary_data = []
    for stock, data in results.items():
        day1 = data['predictions'][0]
        summary_data.append({
            'Stock': stock,
            'Current Price': f"Rs {data['current_price']:,.2f}",
            'Predicted': f"Rs {day1['predicted_price']:,.2f}",
            'Change %': f"{day1['change_pct']:+.2f}%",
            'Direction': day1['direction'],
            'Confidence': f"{day1['confidence']:.1f}%",
            'Model R2': f"{data['cv_r2']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    st.subheader("Portfolio Metrics")
    changes = [results[stock]['predictions'][0]['change_pct'] for stock in results]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Expected Return", f"{np.mean(changes):+.2f}%")
    with col2:
        bullish = sum(1 for c in changes if c > 0)
        st.metric("Bullish Stocks", f"{bullish}/{len(changes)}")
    with col3:
        st.metric("Portfolio Risk", f"{np.std(changes):.2f}%")
    with col4:
        top_stock = max(results.items(), key=lambda x: x[1]['predictions'][0]['change_pct'])
        st.metric("Top Performer", top_stock[0])

def show_home_page():
    st.header("Welcome to Stock Market Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Features
        - Market Indices: NIFTY 50, SENSEX, Bank NIFTY
        - 15+ Stocks: Top Indian companies
        - Multi-day: 1-5 day predictions
        """)
    
    with col2:
        st.markdown("""
        ### Technology
        - ML Algorithm: XGBoost Regressor
        - Indicators: 25+ technical indicators
        - Validation: Time Series Cross-Validation
        - Real-time: Yahoo Finance data
        """)
    
    with col3:
        st.markdown("""
        ### Analytics
        - Sentiment Analysis: News integration
        - Risk Assessment: Confidence scores
        - Visualization: Interactive charts
        - Recommendations: Buy/Sell/Hold
        """)
    
    st.markdown("---")
    st.subheader("Quick Market Overview")
    
    data_handler = st.session_state.data_handler
    indices = {'NIFTY 50': '^NSEI', 'SENSEX': '^BSESN', 'BANK NIFTY': '^NSEBANK'}
    cols = st.columns(len(indices))
    
    for idx, (name, symbol) in enumerate(indices.items()):
        with cols[idx]:
            try:
                data, _, _ = data_handler.fetch_stock_data(symbol, period='5d')
                if data is not None and len(data) >= 2:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100
                    st.metric(name, f"Rs {current_price:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
            except:
                st.metric(name, "Loading...", "")
    
    st.markdown("---")
    st.subheader("Getting Started")
    st.markdown("""
    1. Select Analysis Type from sidebar
    2. Choose Prediction Days (1-5 days)
    3. Click Predict to generate forecasts
    4. Review Results with confidence scores and charts
    """)

def show_nifty_predictor(days):
    st.header("NIFTY 50 Index Predictor")
    data_handler = st.session_state.data_handler
    sentiment_analyzer = st.session_state.sentiment_analyzer
    
    if st.button("Predict NIFTY 50", type="primary", use_container_width=True):
        with st.spinner("Analyzing NIFTY 50 Index..."):
            data, name, sector = data_handler.fetch_stock_data('^NSEI', period='2y')
            if data is None:
                st.error("Failed to fetch NIFTY 50 data")
                return
            
            data = TechnicalIndicators.add_all_indicators(data)
            data = data_handler.clean_data(data)
            sentiment_data = sentiment_analyzer.get_sentiment("NIFTY 50 India")
            
            predictor = StockPredictor()
            X, feature_cols = predictor.prepare_features(data)
            y = predictor.create_target(data, days_ahead=1)
            
            metrics = predictor.train(X, y)
            cv_results = predictor.cross_validate(X, y, n_splits=5)
            predictor.cv_results = cv_results
            predictions = predictor.predict_multi_day(data, days=days)
            
            st.session_state.nifty_prediction = {
                'symbol': '^NSEI', 'name': 'NIFTY 50',
                'current_price': data['Close'].iloc[-1],
                'predictions': predictions, 'sentiment': sentiment_data,
                'metrics': metrics, 'cv_results': cv_results, 'data': data
            }
        
        display_prediction_results(st.session_state.nifty_prediction, predictor)

def show_sensex_predictor(days):
    st.header("SENSEX Index Predictor")
    data_handler = st.session_state.data_handler
    sentiment_analyzer = st.session_state.sentiment_analyzer
    
    if st.button("Predict SENSEX", type="primary", use_container_width=True):
        with st.spinner("Analyzing SENSEX Index..."):
            data, name, sector = data_handler.fetch_stock_data('^BSESN', period='2y')
            if data is None:
                st.error("Failed to fetch SENSEX data")
                return
            
            data = TechnicalIndicators.add_all_indicators(data)
            data = data_handler.clean_data(data)
            sentiment_data = sentiment_analyzer.get_sentiment("SENSEX BSE India")
            
            predictor = StockPredictor()
            X, feature_cols = predictor.prepare_features(data)
            y = predictor.create_target(data, days_ahead=1)
            
            metrics = predictor.train(X, y)
            cv_results = predictor.cross_validate(X, y, n_splits=5)
            predictor.cv_results = cv_results
            predictions = predictor.predict_multi_day(data, days=days)
            
            st.session_state.sensex_prediction = {
                'symbol': '^BSESN', 'name': 'SENSEX',
                'current_price': data['Close'].iloc[-1],
                'predictions': predictions, 'sentiment': sentiment_data,
                'metrics': metrics, 'cv_results': cv_results, 'data': data
            }
        
        display_prediction_results(st.session_state.sensex_prediction, predictor)

def show_stock_predictor(days):
    st.header("Individual Stock Predictor")
    data_handler = st.session_state.data_handler
    sentiment_analyzer = st.session_state.sentiment_analyzer
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_stock = st.selectbox("Select Stock", options=list(data_handler.indian_stocks.keys()),
                                     format_func=lambda x: f"{x} ({data_handler.indian_stocks[x]})")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("Predict", type="primary", use_container_width=True)
    
    if predict_button:
        symbol = data_handler.indian_stocks[selected_stock]
        with st.spinner(f"Analyzing {selected_stock}..."):
            data, company_name, sector = data_handler.fetch_stock_data(symbol, period='2y')
            if data is None:
                st.error(f"Failed to fetch data for {selected_stock}")
                return
            
            data = TechnicalIndicators.add_all_indicators(data)
            data = data_handler.clean_data(data)
            sentiment_data = sentiment_analyzer.get_sentiment(company_name or selected_stock)
            
            predictor = StockPredictor()
            X, feature_cols = predictor.prepare_features(data)
            y = predictor.create_target(data, days_ahead=1)
            
            metrics = predictor.train(X, y)
            cv_results = predictor.cross_validate(X, y, n_splits=5)
            predictor.cv_results = cv_results
            predictions = predictor.predict_multi_day(data, days=days)
            
            prediction_data = {
                'symbol': symbol, 'name': company_name or selected_stock, 'sector': sector,
                'current_price': data['Close'].iloc[-1],
                'predictions': predictions, 'sentiment': sentiment_data,
                'metrics': metrics, 'cv_results': cv_results, 'data': data
            }
            st.session_state[f'stock_{selected_stock}'] = prediction_data
        
        display_prediction_results(prediction_data, predictor)

def show_portfolio_analysis(days):
    st.header("Portfolio Analysis")
    data_handler = st.session_state.data_handler
    
    selected_stocks = st.multiselect("Select Stocks for Portfolio",
        options=list(data_handler.indian_stocks.keys()),
        default=['NIFTY_50', 'SENSEX', 'RELIANCE', 'TCS', 'INFY'],
        format_func=lambda x: f"{x} ({data_handler.indian_stocks[x]})")
    
    if not selected_stocks:
        st.warning("Please select at least one stock")
        return
    
    if st.button("Analyze Portfolio", type="primary", use_container_width=True):
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, stock in enumerate(selected_stocks):
            progress = (idx + 1) / len(selected_stocks)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {stock}... ({idx + 1}/{len(selected_stocks)})")
            
            symbol = data_handler.indian_stocks[stock]
            try:
                data, company_name, sector = data_handler.fetch_stock_data(symbol, period='1y')
                if data is not None:
                    data = TechnicalIndicators.add_all_indicators(data)
                    data = data_handler.clean_data(data)
                    
                    predictor = StockPredictor()
                    X, _ = predictor.prepare_features(data)
                    y = predictor.create_target(data, days_ahead=1)
                    
                    predictor.train(X, y)
                    cv_results = predictor.cross_validate(X, y, n_splits=3)
                    predictor.cv_results = cv_results
                    predictions = predictor.predict_multi_day(data, days=days)
                    
                    results[stock] = {
                        'name': company_name or stock,
                        'current_price': data['Close'].iloc[-1],
                        'predictions': predictions,
                        'cv_r2': cv_results['avg_test_r2']
                    }
            except Exception as e:
                st.warning(f"Could not analyze {stock}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            display_portfolio_results(results)

def show_about_page():
    st.header("About Stock Market Predictor")
    st.markdown("""
    ### What is this?
    AI-powered platform for predicting Indian stock prices and market indices using XGBoost machine learning.
    
    ### How it works:
    1. Data Collection: Real-time data from Yahoo Finance
    2. Feature Engineering: 25+ technical indicators
    3. Machine Learning: XGBoost Regressor with time series cross-validation
    4. Prediction: 1-5 day forecasts with confidence scores
    
    
    ### Algorithm: XGBoost
    Industry-standard gradient boosting algorithm optimized for time series prediction.
    
    ### Disclaimer:
    Not financial advice. Always consult financial advisors.
    
    """)

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Choose Analysis:", [
        "Home", "NIFTY 50 Predictor", "SENSEX Predictor",
        "Individual Stocks", "Portfolio Analysis", "About"
    ], index=0)
    
    st.markdown("---")
    st.markdown("### Settings")
    prediction_days = st.slider("Prediction Days", 1, 5, 3)
    
    st.markdown("---")
    st.info(f"\n\n{datetime.now().strftime('%d %B %Y')}\n\n{datetime.now().strftime('%H:%M:%S')}")

if page == "Home":
    show_home_page()
elif page == "NIFTY 50 Predictor":
    show_nifty_predictor(prediction_days)
elif page == "SENSEX Predictor":
    show_sensex_predictor(prediction_days)
elif page == "Individual Stocks":
    show_stock_predictor(prediction_days)
elif page == "Portfolio Analysis":
    show_portfolio_analysis(prediction_days)
else:
    show_about_page()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Stock Market Predictor v1.0 </p>
    <p>Data by Yahoo Finance | Not Financial Advice</p>
</div>
""", unsafe_allow_html=True)