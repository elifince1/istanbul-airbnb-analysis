# Data Directory

## Structure

```
data/
├── raw/                  # Downloaded from Inside Airbnb (not committed)
│   ├── listings.csv.gz   # Detailed listing info (~30K rows × 79 cols)
│   ├── calendar.csv.gz   # Daily availability (~11M rows)
│   ├── reviews.csv.gz    # Individual reviews (~545K rows)
│   └── neighbourhoods.csv
└── processed/
    └── listings_enriched.csv  # Final analysis dataset (one row per listing)
```

## Data Source

- **Source:** [Inside Airbnb – Istanbul](http://insideairbnb.com/get-the-data/)
- **Snapshot date:** September 29, 2025
- **License:** Creative Commons Attribution 4.0

## How to Reproduce

```bash
# 1. Download raw data
python src/download_data.py

# 2. Run preprocessing pipeline
python src/preprocess.py
```

Both raw and processed CSV files are excluded from version control by default. They can be recreated locally using the commands above.

## Preprocessing Steps

1. **Price cleaning** — Removed `$` and `,` symbols, converted to float
2. **Boolean encoding** — Converted `host_is_superhost`, `instant_bookable`, `has_availability` from `t`/`f` to `True`/`False`
3. **Percentage columns** — Converted `host_response_rate` and `host_acceptance_rate` to numeric
4. **Bathrooms** — Extracted numeric values from `bathrooms_text` (e.g. "1 bath" → 1.0)
5. **Calendar aggregation** — Aggregated daily calendar data to listing level (note: calendar prices were empty in this snapshot)
6. **Review aggregation** — Computed `review_count_computed`, `first_review_computed`, `last_review_computed`, `review_span_days`, `reviews_per_month_computed` from individual review records
7. **Neighbourhood audit** — Downloaded `neighbourhoods.csv` as a reference file and confirmed that the cleaned listing file already contains the neighbourhood labels used in the final analysis
8. **Missing values** — Dropped columns with >80% missing values (4 columns removed)
9. **Price filtering** — Removed listings with missing, zero, or extremely high (>100K TRY) prices
10. **Merge** — Joined listings, calendar aggregates, and review aggregates on listing ID

## Final Dataset Summary

| Property | Value |
|----------|-------|
| Rows (listings) | ~25,206 |
| Columns (features) | 81 |
| Price range | 80 – 100,000 TRY |
| Median price | ~2,535 TRY |

## Key Variables

| Column | Type | Description |
|--------|------|-------------|
| `price` | float | Listing price in TRY (cleaned) |
| `room_type` | str | Entire home/apt, Private room, Shared room, Hotel room |
| `neighbourhood_cleansed` | str | Istanbul neighbourhood |
| `host_is_superhost` | bool | Whether the host is a superhost |
| `accommodates` | int | Maximum number of guests |
| `bedrooms` | float | Number of bedrooms |
| `beds` | float | Number of beds |
| `bathrooms` | float | Number of bathrooms (extracted from text) |
| `review_scores_rating` | float | Overall review score (0–5) |
| `number_of_reviews` | int | Total number of reviews |
| `reviews_per_month` | float | Average reviews per month |
| `availability_30/60/365` | int | Days available in next 30/60/365 days |
| `minimum_nights` | int | Minimum night stay requirement |
| `instant_bookable` | bool | Whether instant booking is enabled |
| `latitude`, `longitude` | float | Geographic coordinates |

## Notes

- Raw and processed data files are **not committed** to this repository by default (excluded via `.gitignore`).
- Run `python src/download_data.py` to obtain the raw files (~106 MB total download).
- Run `python src/preprocess.py` to recreate the processed listing-level dataset.
