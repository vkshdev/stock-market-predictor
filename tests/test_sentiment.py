from sentiment import SentimentAnalyzer


def test_sentiment_is_explicitly_unavailable_without_api_key() -> None:
    result = SentimentAnalyzer().get_sentiment("Reliance Industries")

    assert result["available"] is False
    assert result["sentiment_trend"] == "Unavailable"
    assert result["avg_sentiment"] is None
    assert result["article_count"] == 0
    assert result["articles"] == []
    assert "NEWS_API_KEY" in result["reason"]
