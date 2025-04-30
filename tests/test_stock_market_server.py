import json
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import stock_market_server

@pytest.fixture
def mock_finnhub_client():
    """Create a mock Finnhub client for testing."""
    mock_client = MagicMock()
    return mock_client

@patch('stock_market_server.finnhub_client')
def test_get_stock_symbol_lookup(mock_client):
    """Test the get_stock_symbol_lookup function."""
    # Setup mock response
    mock_client.symbol_lookup.return_value = {
        "count": 2,
        "result": [
            {
                "description": "APPLE INC",
                "displaySymbol": "AAPL",
                "symbol": "AAPL",
                "type": "Common Stock"
            },
            {
                "description": "APPLE INC",
                "displaySymbol": "AAPL.SW",
                "symbol": "AAPL.SW",
                "type": "Common Stock"
            }
        ]
    }
    
    # Call the function
    result = stock_market_server.get_stock_symbol_lookup("apple")
    
    # Verify the result
    result_dict = json.loads(result)
    assert result_dict["search_query"] == "apple"
    assert result_dict["total_matches"] == 2
    assert len(result_dict["matches"]) == 2
    assert result_dict["matches"][0]["symbol"] == "AAPL"

@patch('stock_market_server.finnhub_client')
def test_get_stock_price(mock_client):
    """Test the get_stock_price function."""
    # Setup mock response
    mock_client.quote.return_value = {
        "c": 150.25,
        "h": 152.30,
        "l": 149.50,
        "o": 151.00,
        "pc": 148.75,
        "t": 1682345678
    }
    
    # Call the function
    result = stock_market_server.get_stock_price("AAPL")
    
    # Verify the result
    result_dict = json.loads(result)
    assert result_dict["symbol"] == "AAPL"
    assert result_dict["current_price"] == 150.25
    assert result_dict["day_high"] == 152.30

@patch('stock_market_server.finnhub_client')
def test_get_basic_financials(mock_client):
    """Test the get_basic_financials function."""
    # Setup mock response
    mock_client.company_basic_financials.return_value = {
        "metric": {
            "marketCapitalization": 2500000000000,
            "peBasicExclExtraTTM": 28.5,
            "pbQuarterlyTTM": 15.2,
            "dividendYieldIndicatedAnnual": 0.65
        },
        "series": {
            "name": "Apple Inc"
        }
    }
    
    # Call the function
    result = stock_market_server.get_basic_financials("AAPL")
    
    # Verify the result
    result_dict = json.loads(result)
    assert result_dict["symbol"] == "AAPL"
    assert result_dict["company_name"] == "Apple Inc"
    assert result_dict["pe_ratio"] == 28.5
    assert result_dict["dividend_yield"] == 0.65

@patch('stock_market_server.finnhub_client')
def test_get_market_news(mock_client):
    """Test the get_market_news function."""
    # Setup mock response
    mock_client.general_news.return_value = [
        {
            "id": 12345,
            "headline": "Market Update: Stocks Rise",
            "summary": "Major indices are up today.",
            "source": "Financial News",
            "datetime": 1682345678,
            "url": "https://example.com/news/12345",
            "related": ["AAPL", "MSFT"]
        }
    ]
    
    # Call the function
    result = stock_market_server.get_market_news("general")
    
    # Verify the result
    result_dict = json.loads(result)
    assert result_dict["category"] == "general"
    assert result_dict["news_count"] == 1
    assert len(result_dict["articles"]) == 1
    assert result_dict["articles"][0]["headline"] == "Market Update: Stocks Rise"

@patch('stock_market_server.finnhub_client')
def test_get_company_news(mock_client):
    """Test the get_company_news function."""
    # Setup mock response
    mock_client.company_news.return_value = [
        {
            "headline": "Apple Announces New iPhone",
            "summary": "Apple unveils the latest iPhone model.",
            "source": "Tech News",
            "datetime": 1682345678,
            "url": "https://example.com/news/apple-iphone",
            "related": ["AAPL"]
        }
    ]
    
    # Call the function
    result = stock_market_server.get_company_news("AAPL", "2023-01-01", "2023-01-31")
    
    # Verify the result
    result_dict = json.loads(result)
    assert result_dict["symbol"] == "AAPL"
    assert result_dict["from_date"] == "2023-01-01"
    assert result_dict["to_date"] == "2023-01-31"
    assert result_dict["news_count"] == 1
    assert result_dict["articles"][0]["headline"] == "Apple Announces New iPhone"

@patch('stock_market_server.finnhub_client')
@patch('stock_market_server.time.time')
def test_get_stock_candles(mock_time, mock_client):
    """Test the get_stock_candles function."""
    # Setup mock time
    mock_time.return_value = 1682345678
    
    # Setup mock response
    mock_client.stock_candles.return_value = {
        "s": "ok",
        "t": [1682259278, 1682345678],
        "o": [150.0, 151.0],
        "h": [152.0, 153.0],
        "l": [149.0, 150.0],
        "c": [151.0, 152.0],
        "v": [1000000, 1200000]
    }
    
    # Call the function
    result = stock_market_server.get_stock_candles("AAPL", "D")
    
    # Verify the result
    result_dict = json.loads(result)
    assert result_dict["symbol"] == "AAPL"
    assert result_dict["resolution"] == "D"
    assert result_dict["status"] == "ok"
    assert result_dict["candle_count"] == 2
    assert len(result_dict["candles"]) == 2
    assert result_dict["candles"][0]["open"] == 150.0
    assert result_dict["candles"][1]["close"] == 152.0
