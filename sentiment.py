"""Optional timestamped news sentiment analysis."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

import numpy as np
import requests
from textblob import TextBlob


logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze real news when a provider key is configured."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def get_sentiment(self, company_name: str, days: int = 7) -> dict[str, object]:
        if not self.api_key:
            return self._unavailable("NEWS_API_KEY is not configured")
        return self._fetch_real_sentiment(company_name, days)

    def _fetch_real_sentiment(
        self,
        company_name: str,
        days: int,
    ) -> dict[str, object]:
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": company_name,
                    "from": (
                        datetime.now(timezone.utc) - timedelta(days=days)
                    ).strftime("%Y-%m-%d"),
                    "sortBy": "publishedAt",
                    "apiKey": self.api_key,
                    "language": "en",
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "ok":
                return self._unavailable(
                    str(payload.get("message", "News provider returned an error"))
                )

            articles = payload.get("articles", [])
            if not articles:
                return self._unavailable("No recent articles were returned")
            return self._analyze_articles(articles)
        except (requests.RequestException, ValueError) as error:
            logger.warning("News sentiment unavailable: %s", error)
            return self._unavailable("News provider request failed")

    def _analyze_articles(self, articles: list[dict[str, object]]) -> dict[str, object]:
        scored_articles: list[dict[str, object]] = []

        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".strip()
            if not text:
                continue
            sentiment = TextBlob(text).sentiment
            scored_articles.append(
                {
                    "title": article.get("title"),
                    "source": (article.get("source") or {}).get("name"),
                    "sentiment": float(sentiment.polarity),
                    "subjectivity": float(sentiment.subjectivity),
                    "published_at": article.get("publishedAt"),
                }
            )

        if not scored_articles:
            return self._unavailable("Articles did not contain scorable text")

        average_sentiment = float(
            np.mean([article["sentiment"] for article in scored_articles])
        )
        average_subjectivity = float(
            np.mean([article["subjectivity"] for article in scored_articles])
        )

        return {
            "available": True,
            "reason": None,
            "avg_sentiment": average_sentiment,
            "sentiment_trend": self._classify_sentiment(average_sentiment),
            "average_subjectivity": average_subjectivity,
            "article_count": len(scored_articles),
            "articles": scored_articles[:5],
        }

    @staticmethod
    def _unavailable(reason: str) -> dict[str, object]:
        return {
            "available": False,
            "reason": reason,
            "avg_sentiment": None,
            "sentiment_trend": "Unavailable",
            "average_subjectivity": None,
            "article_count": 0,
            "articles": [],
        }

    @staticmethod
    def _classify_sentiment(score: float) -> str:
        if score > 0.1:
            return "Positive"
        if score < -0.1:
            return "Negative"
        return "Neutral"
