"""
Data fetching and preprocessing module
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataHandler:
    
    def __init__(self):
        self.indian_stocks = {
            'NIFTY_50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY_BANK': '^NSEBANK',
            'NIFTY_IT': '^CNXIT',
            'NIFTY_PHARMA': '^CNXPHARMA',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFY': 'INFY.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'SBIN': 'SBIN.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'ASIANPAINT': 'ASIANPAINT.NS',
            'WIPRO': 'WIPRO.NS',
            'HCLTECH': 'HCLTECH.NS',
            'MARUTI': 'MARUTI.NS',
            'NTPC': 'NTPC.NS',
            'ADANIPORTS': 'ADANIPORTS.NS'
        }
    
    def fetch_stock_data(self, symbol, period='5y', interval='1d'):
        """Fetch stock data - now defaults to 5 years"""
        try:
            logger.info(f"Fetching {period} data for {symbol}")
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.error(f"No data found for {symbol}")
                return None, None, None
            
            info = stock.info
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'Unknown')
            
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            
            return data, company_name, sector
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None, None, None
    
    def clean_data(self, df):
        """Clean and preprocess stock data"""
        df = df.dropna()
        df = df.ffill().bfill()
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        return df
    
    def get_latest_price(self, symbol):
        """Get latest price for a symbol"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period='1d')
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except:
            return None