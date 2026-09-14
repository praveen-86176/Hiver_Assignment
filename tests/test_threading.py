"""Unit tests for thread reconstruction logic in src/data/threading.py."""

import pandas as pd
import pytest
from src.data.threading import identify_brand_accounts, reconstruct_pairs


@pytest.fixture
def sample_toy_df() -> pd.DataFrame:
    """Create a hand-crafted DataFrame with standard, multi-reply, multi-turn, and unanswered cases."""
    data = [
        # Conversation 1: Standard Customer -> Brand pair (AppleSupport)
        {
            "tweet_id": 101,
            "author_id": "cust_1",
            "inbound": True,
            "created_at": "2023-01-01 10:00:00",
            "text": "My iPhone battery is draining very fast.",
            "response_tweet_id": "201",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": 201,
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "2023-01-01 10:05:00",
            "text": "@cust_1 We'd love to help! Please check Settings > Battery.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 101,
        },
        # Conversation 2: Customer tweet with NO brand reply (Unanswered inquiry)
        {
            "tweet_id": 102,
            "author_id": "cust_2",
            "inbound": True,
            "created_at": "2023-01-01 11:00:00",
            "text": "@AppleSupport I lost my AirPods case in the cafe.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": None,
        },
        # Conversation 3: Multiple brand replies to the same customer tweet (Multi-reply)
        {
            "tweet_id": 103,
            "author_id": "cust_3",
            "inbound": True,
            "created_at": "2023-01-01 12:00:00",
            "text": "@SpotifyCares Can't download songs offline.",
            "response_tweet_id": "203,204",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": 203,
            "author_id": "SpotifyCares",
            "inbound": False,
            "created_at": "2023-01-01 12:04:00",  # Earliest reply
            "text": "@cust_3 Let's help! Which device OS and Spotify version are you using?",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 103,
        },
        {
            "tweet_id": 204,
            "author_id": "SpotifyCares",
            "inbound": False,
            "created_at": "2023-01-01 12:10:00",  # Later follow-up reply
            "text": "@cust_3 Also make sure you have available storage space.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 103,
        },
        # Conversation 4: Multi-turn continuation (Customer replies back to brand)
        {
            "tweet_id": 104,
            "author_id": "cust_1",
            "inbound": True,
            "created_at": "2023-01-01 10:10:00",
            "text": "@AppleSupport Battery health says 78% Service.",
            "response_tweet_id": "205",
            "in_response_to_tweet_id": 201,  # In response to previous brand tweet
        },
        {
            "tweet_id": 205,
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "2023-01-01 10:15:00",
            "text": "@cust_1 You can schedule a battery replacement appointment at an Apple Store.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 104,
        },
    ]
    return pd.DataFrame(data)


def test_standard_pairing(sample_toy_df: pd.DataFrame):
    """Test standard customer -> brand pairing reconstructs expected fields."""
    pairs = reconstruct_pairs(sample_toy_df, brand="AppleSupport")
    
    # In initial inquiry MVP mode, only tweet 101 should be paired for AppleSupport
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["thread_id"] == 101
    assert row["customer_message"] == "My iPhone battery is draining very fast."
    assert "Settings > Battery" in row["brand_reply"]
    assert row["brand"] == "AppleSupport"


def test_unanswered_customer_tweet_excluded(sample_toy_df: pd.DataFrame):
    """Edge Case: Customer tweet with no matching reply should be safely excluded without crash."""
    pairs = reconstruct_pairs(sample_toy_df)
    
    # Tweet 102 has no reply, so it must not appear in reconstructed pairs
    thread_ids = set(pairs["thread_id"].tolist())
    assert 102 not in thread_ids
    assert len(pairs) == 2  # Tweet 101 (Apple) and Tweet 103 (Spotify)


def test_multiple_replies_picks_earliest(sample_toy_df: pd.DataFrame):
    """Test that multiple brand replies to the same customer tweet resolves to the earliest reply."""
    pairs = reconstruct_pairs(sample_toy_df, brand="SpotifyCares")
    
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["thread_id"] == 103
    # Must pick tweet 203 (12:04:00), not tweet 204 (12:10:00)
    assert "Which device OS and Spotify version" in row["brand_reply"]
    assert "storage space" not in row["brand_reply"]


def test_multi_turn_handling_mode(sample_toy_df: pd.DataFrame):
    """Test behavior with require_initial_inquiry=False vs True."""
    # When require_initial_inquiry=False, tweet 104 (follow-up) is also included
    pairs_all = reconstruct_pairs(sample_toy_df, brand="AppleSupport", require_initial_inquiry=False)
    assert len(pairs_all) == 2
    assert set(pairs_all["thread_id"].tolist()) == {101, 104}

    # When require_initial_inquiry=True (MVP default), only initial inquiry 101 is included
    pairs_mvp = reconstruct_pairs(sample_toy_df, brand="AppleSupport", require_initial_inquiry=True)
    assert len(pairs_mvp) == 1
    assert pairs_mvp.iloc[0]["thread_id"] == 101


def test_identify_brand_accounts(sample_toy_df: pd.DataFrame):
    """Test brand accounts identification."""
    brands = identify_brand_accounts(sample_toy_df)
    assert brands == {"AppleSupport", "SpotifyCares"}


def test_empty_dataframe():
    """Test reconstruct_pairs on empty DataFrame."""
    empty_df = pd.DataFrame(columns=["tweet_id", "author_id", "inbound", "created_at", "text", "in_response_to_tweet_id"])
    pairs = reconstruct_pairs(empty_df)
    assert isinstance(pairs, pd.DataFrame)
    assert len(pairs) == 0
    assert "thread_id" in pairs.columns
    assert "customer_message" in pairs.columns
    assert "brand_reply" in pairs.columns
