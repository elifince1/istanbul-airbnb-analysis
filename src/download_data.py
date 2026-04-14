#!/usr/bin/env python3
"""Download Istanbul Airbnb data from Inside Airbnb (Sept 29, 2025 snapshot)."""

import os
import requests

BASE_URL = "https://data.insideairbnb.com/turkey/marmara/istanbul/2025-09-29"

FILES = {
    "listings.csv.gz": f"{BASE_URL}/data/listings.csv.gz",
    "calendar.csv.gz": f"{BASE_URL}/data/calendar.csv.gz",
    "reviews.csv.gz":  f"{BASE_URL}/data/reviews.csv.gz",
    "neighbourhoods.csv": f"{BASE_URL}/visualisations/neighbourhoods.csv",
}

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def download_file(url, filepath):
    """Download a file from url to filepath with progress feedback."""
    print(f"  Downloading {os.path.basename(filepath)} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved ({size_mb:.1f} MB)")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Target directory: {RAW_DIR}\n")

    for filename, url in FILES.items():
        filepath = os.path.join(RAW_DIR, filename)
        if os.path.exists(filepath):
            print(f"  {filename} already exists, skipping.")
        else:
            download_file(url, filepath)

    print("\nAll files downloaded successfully.")
    for f in os.listdir(RAW_DIR):
        size = os.path.getsize(os.path.join(RAW_DIR, f)) / (1024 * 1024)
        print(f"  {f}: {size:.1f} MB")


if __name__ == "__main__":
    main()
