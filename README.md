# Istanbul Airbnb Listings Analysis

## Project Overview

This is a term project for **DSA 210 – Introduction to Data Science** (Spring 2025–2026). The goal is to analyze Istanbul Airbnb listings to understand which features most strongly influence listing price and demand. By examining publicly available data from Inside Airbnb and enriching it with additional contextual information, this project aims to uncover actionable patterns in Istanbul's short-term rental market.

## Research Question

**Which listing features — such as location, property type, host characteristics, and review patterns — are the strongest predictors of price and occupancy for Airbnb rentals in Istanbul?**

## Data Source

- **Primary dataset:** [Inside Airbnb – Istanbul](http://insideairbnb.com/get-the-data/) — publicly available listing, review, and calendar data.
- **Enrichment (planned):** Additional contextual data such as neighborhood-level attributes or tourism indicators to complement the primary dataset.

## Motivation

Istanbul is one of the most visited cities in the world, with a rapidly growing short-term rental market. Understanding the factors that drive pricing and demand can provide insights for hosts, travelers, and urban policymakers alike. This project applies data science methods to a tangible, local problem using real-world data.

## Project Roadmap

| Milestone | Deadline | Status |
|-----------|----------|--------|
| Repository creation | March 17 | ✅ Complete |
| Project proposal | March 31 | Upcoming |
| Data collection, EDA & hypothesis testing | April 14 | Upcoming |
| Machine learning methods | May 5 | Upcoming |
| Final report & code submission | May 18 | Upcoming |

## Reproducibility

This repository will be expanded incrementally throughout the semester. Each milestone will add new data, analysis, and documentation. To reproduce the analysis at any stage:

1. Clone this repository.
2. Create a virtual environment: `python3 -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Follow instructions in the relevant notebook(s) under `notebooks/`.

## Repository Structure

```
├── data/              # Datasets (raw data not committed; see data/README.md)
├── notebooks/         # Jupyter notebooks for analysis
├── src/               # Python source code and utilities
├── reports/           # Written reports and deliverables
├── figures/           # Generated plots and visualizations
├── requirements.txt   # Python dependencies
└── README.md          # This file
```
