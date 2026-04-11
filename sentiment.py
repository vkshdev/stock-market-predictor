"""
News sentiment analysis module
"""

import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import numpy as np

class SentimentAnalyzer:
    """Analyze news sentiment for stocks"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def get_sentiment(self, company_name, days=7):
        """
        Get sentiment score for a company
        
        Args:
            company_name (str): Company name
            days (int): Number of days to analyze
        
        Returns:
            dict: Sentiment analysis results
        """
        if self.api_key:
            return self._fetch_real_sentiment(company_name, days)
        else:
            return self._generate_mock_sentiment(company_name, days)
    
    def _fetch_real_sentiment(self, company_name, days):
        """Fetch real news and analyze sentiment (NewsAPI)"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': company_name,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'apiKey': self.api_key,
                'language': 'en'
            }
            
            response = requests.get(url, params=params, timeout=10)
            articles = response.json().get('articles', [])
            
            return self._analyze_articles(articles)
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self._generate_mock_sentiment(company_name, days)
    
    def _analyze_articles(self, articles):
        """Analyze sentiment of articles"""
        sentiments = []
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            blob = TextBlob(text)
            
            sentiments.append({
                'title': article.get('title'),
                'sentiment': blob.sentiment.polarity,
                'confidence': blob.sentiment.subjectivity,
                'date': article.get('publishedAt')
            })
        
        avg_sentiment = np.mean([s['sentiment'] for s in sentiments]) if sentiments else 0
        
        return {
            'avg_sentiment': avg_sentiment,
            'sentiment_trend': self._classify_sentiment(avg_sentiment),
            'confidence': np.mean([s['confidence'] for s in sentiments]) if sentiments else 0.5,
            'article_count': len(sentiments),
            'articles': sentiments[:5]
        }
    
    def _generate_mock_sentiment(self, company_name, days):
        """Generate mock sentiment for testing"""
        np.random.seed(hash(company_name) % 2**32)
        sentiment_score = np.random.uniform(-0.5, 0.5)
        
        return {
            'avg_sentiment': sentiment_score,
            'sentiment_trend': self._classify_sentiment(sentiment_score),
            'confidence': np.random.uniform(0.6, 0.9),
            'article_count': np.random.randint(5, 20),
            'articles': []
        }
    
    @staticmethod
    def _classify_sentiment(score):
        """Classify sentiment score"""
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'