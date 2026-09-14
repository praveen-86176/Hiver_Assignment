"""Thread reconstruction module for Customer Support conversations.

Reconstructs customer -> brand reply pairs from raw Twitter customer support data.
Identifies initial customer inquiries and pairs them with the brand's first substantive reply.
"""

from typing import List, Optional, Set
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def identify_brand_accounts(df: pd.DataFrame) -> Set[str]:
    """Identify all brand account handles in the dataset based on outbound tweets.

    Args:
        df: Input DataFrame containing 'inbound' and 'author_id' columns.

    Returns:
        Set of string brand handles (author_ids where inbound is False).
    """
    if "inbound" not in df.columns or "author_id" not in df.columns:
        raise ValueError("DataFrame must contain 'inbound' and 'author_id' columns.")
    
    brand_handles = set(df[df["inbound"] == False]["author_id"].dropna().unique())
    return brand_handles


def reconstruct_pairs(
    df: pd.DataFrame,
    brand: Optional[str] = None,
    require_initial_inquiry: bool = True,
) -> pd.DataFrame:
    """Reconstruct (customer_message, brand_reply) pairs from conversation data.

    MVP Scope:
    - Customer message: The initial customer tweet starting the thread
      (inbound=True, in_response_to_tweet_id is null/NaN).
    - Brand reply: The brand's first substantive reply to that customer tweet
      (inbound=False, in_response_to_tweet_id == customer tweet_id).
    
    # TODO(phase2): Implement full multi-turn thread reconstruction for conversations
    # going 3+ turns deep (e.g. customer -> brand -> customer -> brand). In multi-turn
    # contexts, recursive graph traversal or windowed sessionization will capture ongoing
    # resolution dialogs and follow-up exchanges beyond the initial response pair.

    Args:
        df: Input DataFrame with raw/cleaned TWCS columns.
        brand: Optional brand author_id to filter for (e.g. 'AppleSupport', 'AmazonHelp').
               If None, reconstructs pairs across all brands.
        require_initial_inquiry: If True, only pairs where customer tweet is the thread
                                 starter (in_response_to_tweet_id is NaN) are kept.

    Returns:
        DataFrame with columns:
            - thread_id: Identifier for the thread (initial customer tweet_id)
            - customer_message: Raw text of customer tweet
            - brand_reply: Raw text of brand reply
            - customer_ts: Timestamp of customer tweet
            - brand_ts: Timestamp of brand reply
            - brand: Brand handle (author_id)
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "thread_id",
                "customer_message",
                "brand_reply",
                "customer_ts",
                "brand_ts",
                "brand",
            ]
        )

    # 1. Separate inbound (customer) and outbound (brand)
    inbound_df = df[df["inbound"] == True].copy()
    outbound_df = df[df["inbound"] == False].copy()

    # Filter by brand if specified
    if brand is not None:
        outbound_df = outbound_df[outbound_df["author_id"].astype(str).str.lower() == str(brand).lower()].copy()

    # 2. Filter customer inquiries
    if require_initial_inquiry:
        customer_candidates = inbound_df[inbound_df["in_response_to_tweet_id"].isna()].copy()
    else:
        customer_candidates = inbound_df.copy()

    # 3. Filter brand replies (must reply to some tweet)
    brand_replies = outbound_df.dropna(subset=["in_response_to_tweet_id"]).copy()
    if brand_replies.empty or customer_candidates.empty:
        return pd.DataFrame(
            columns=[
                "thread_id",
                "customer_message",
                "brand_reply",
                "customer_ts",
                "brand_ts",
                "brand",
            ]
        )

    # Normalize response IDs to integer/numeric for reliable merging
    brand_replies["in_response_to_tweet_id"] = pd.to_numeric(
        brand_replies["in_response_to_tweet_id"], errors="coerce"
    )
    customer_candidates["tweet_id"] = pd.to_numeric(
        customer_candidates["tweet_id"], errors="coerce"
    )

    brand_replies = brand_replies.dropna(subset=["in_response_to_tweet_id"])
    customer_candidates = customer_candidates.dropna(subset=["tweet_id"])

    brand_replies["in_response_to_tweet_id"] = brand_replies["in_response_to_tweet_id"].astype("int64")
    customer_candidates["tweet_id"] = customer_candidates["tweet_id"].astype("int64")

    # 4. Sort brand replies to pick the first reply per customer tweet
    if "created_at" in brand_replies.columns:
        brand_replies = brand_replies.sort_values(
            by=["in_response_to_tweet_id", "created_at", "tweet_id"]
        )
    else:
        brand_replies = brand_replies.sort_values(
            by=["in_response_to_tweet_id", "tweet_id"]
        )

    # Keep first substantive reply for MVP pair
    first_brand_replies = brand_replies.drop_duplicates(
        subset=["in_response_to_tweet_id"], keep="first"
    )

    # 5. Inner join customer messages with brand replies
    # Unanswered customer tweets are naturally excluded by inner join
    paired = first_brand_replies.merge(
        customer_candidates,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_brand", "_customer"),
        how="inner",
    )

    # 6. Format output DataFrame
    output_df = pd.DataFrame(
        {
            "thread_id": paired["tweet_id_customer"],
            "customer_message": paired["text_customer"],
            "brand_reply": paired["text_brand"],
            "customer_ts": paired["created_at_customer"],
            "brand_ts": paired["created_at_brand"],
            "brand": paired["author_id_brand"],
        }
    )

    logger.info(
        "Reconstructed %s pairs%s.",
        f"{len(output_df):,}",
        f" for brand '{brand}'" if brand else " across all brands",
    )
    return output_df
