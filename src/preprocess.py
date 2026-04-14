#!/usr/bin/env python3
"""
Preprocess Istanbul Airbnb data into a single listing-level analysis dataset.

Reads raw files from data/raw/, cleans and merges them, and writes the
enriched dataset to data/processed/listings_enriched.csv.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "listings_enriched.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_price(series):
    """Remove $ and , from price strings and convert to float."""
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace(["", "nan"], np.nan)
        .astype(float)
    )


# ---------------------------------------------------------------------------
# Load & clean listings
# ---------------------------------------------------------------------------

def load_listings():
    print("Loading listings ...")
    df = pd.read_csv(os.path.join(RAW_DIR, "listings.csv.gz"), low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Price
    df["price"] = clean_price(df["price"])

    # Booleans
    bool_map = {"t": True, "f": False}
    for col in ["host_is_superhost", "instant_bookable", "has_availability"]:
        if col in df.columns:
            df[col] = df[col].map(bool_map)

    # Percentage columns → numeric
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace("%", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date columns
    for col in ["host_since", "first_review", "last_review"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Extract numeric bathrooms from bathrooms_text (e.g. "1 bath" → 1.0)
    if "bathrooms_text" in df.columns:
        df["bathrooms"] = (
            df["bathrooms_text"]
            .astype(str)
            .str.extract(r"([\d.]+)", expand=False)
            .astype(float)
        )

    return df


# ---------------------------------------------------------------------------
# Aggregate calendar
# ---------------------------------------------------------------------------

def aggregate_calendar():
    print("Loading and aggregating calendar ...")
    cal = pd.read_csv(os.path.join(RAW_DIR, "calendar.csv.gz"))
    print(f"  Raw calendar shape: {cal.shape}")

    cal["price"] = clean_price(cal["price"])
    cal["available"] = cal["available"].map({"t": True, "f": False})
    cal["date"] = pd.to_datetime(cal["date"])

    min_date = cal["date"].min()
    cal["day_offset"] = (cal["date"] - min_date).dt.days

    # Full-year aggregation
    agg = cal.groupby("listing_id").agg(
        avg_calendar_price=("price", "mean"),
    ).reset_index()

    # Short-window availability (listing-level availability already exists
    # in the listings file, but avg_calendar_price is new)
    print(f"  Aggregated calendar shape: {agg.shape}")
    return agg


# ---------------------------------------------------------------------------
# Aggregate reviews
# ---------------------------------------------------------------------------

def aggregate_reviews():
    print("Loading and aggregating reviews ...")
    rev = pd.read_csv(os.path.join(RAW_DIR, "reviews.csv.gz"))
    print(f"  Raw reviews shape: {rev.shape}")

    rev["date"] = pd.to_datetime(rev["date"])

    agg = rev.groupby("listing_id").agg(
        review_count_computed=("id", "count"),
        first_review_computed=("date", "min"),
        last_review_computed=("date", "max"),
    ).reset_index()

    agg["review_span_days"] = (
        agg["last_review_computed"] - agg["first_review_computed"]
    ).dt.days

    agg["reviews_per_month_computed"] = np.where(
        agg["review_span_days"] > 0,
        agg["review_count_computed"] / (agg["review_span_days"] / 30.44),
        np.nan,
    )

    print(f"  Aggregated reviews shape: {agg.shape}")
    return agg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    listings = load_listings()
    cal_agg = aggregate_calendar()
    rev_agg = aggregate_reviews()

    # Merge
    print("\nMerging datasets ...")
    df = listings.merge(cal_agg, left_on="id", right_on="listing_id",
                        how="left", suffixes=("", "_cal"))
    df = df.merge(rev_agg, left_on="id", right_on="listing_id",
                  how="left", suffixes=("", "_rev"))

    # Drop redundant merge keys
    for col in ["listing_id", "listing_id_rev"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Missing-value report
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
    print(f"\nColumns with >50% missing ({len(high_missing)}):")
    for col, pct in high_missing.items():
        print(f"  {col}: {pct:.1f}%")

    # Drop columns with >80% missing
    drop_cols = missing_pct[missing_pct > 80].index.tolist()
    if drop_cols:
        print(f"\nDropping {len(drop_cols)} columns with >80% missing values")
        df.drop(columns=drop_cols, inplace=True)

    # Remove rows with missing or non-positive price
    before = len(df)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    print(f"\nDropped {before - len(df)} listings with missing/non-positive price")

    # Cap extreme prices at 100 000 TRY
    extreme = (df["price"] > 100_000).sum()
    if extreme:
        print(f"Dropping {extreme} listings with price > 100,000 TRY")
        df = df[df["price"] <= 100_000]

    print(f"\nFinal dataset shape: {df.shape}")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    main()
