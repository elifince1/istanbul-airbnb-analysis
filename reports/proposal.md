# Project Proposal — Istanbul Airbnb Analysis

**Course:** DSA 210 Introduction to Data Science — Spring 2025–2026  
**Student:** Elif İnce

## Data Source

The primary data source for this project is [Inside Airbnb](http://insideairbnb.com/get-the-data/), a publicly available initiative that provides periodic snapshots of Airbnb listing activity for cities worldwide. I will use the Istanbul dataset, specifically the snapshot dated **September 29, 2025**.

## Data Collection

The data will be downloaded directly from the Inside Airbnb website as compressed CSV files. No web scraping or API access is required; the files are openly available for download. The Istanbul dataset consists of four related files:

- **listings.csv.gz** — Detailed listing-level information including host attributes, location, pricing, property features, review scores, and availability.
- **calendar.csv.gz** — Daily availability and price records for each listing over the upcoming year.
- **reviews.csv.gz** — Individual review records with reviewer IDs and dates.
- **neighbourhoods.csv** — A reference table mapping each listing to its neighbourhood within Istanbul.

To enrich the analysis beyond a single flat dataset, I will combine these four files at the listing level. From the calendar data, I will compute availability summaries (e.g., the proportion of days available in the next 30, 60, and 365 days). From the reviews data, I will derive review frequency and recency features. The neighbourhoods file will provide spatial grouping for comparative analysis across Istanbul's districts.

## Dataset Characteristics

The detailed listings file contains approximately **30,300 listings** described by **79 features** covering host information (e.g., host response rate, superhost status), property attributes (e.g., room type, accommodates, bathrooms), location (neighbourhood, coordinates), pricing, review scores across multiple dimensions, and availability metrics. The calendar file contains roughly **11 million daily records** (one per listing per day for 365 days), and the reviews file contains the full text of hundreds of thousands of individual reviews. After merging and feature engineering, the final analysis dataset will have one row per listing with an enriched set of features derived from all four source files.
