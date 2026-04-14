# Istanbul Airbnb Analysis

This repository contains my DSA 210 (Introduction to Data Science) term project for Spring 2025-2026. The project studies how listing, host, and review-related features are associated with Airbnb price and availability patterns in Istanbul.

## Research Question

**Which features of an Airbnb listing are most strongly associated with its price in Istanbul?**

Supporting questions:

- How do room type, neighbourhood, and host status relate to price?
- How do room type and host characteristics relate to availability patterns?
- Are review-related signals associated with higher listing prices?

## Data Source

The project uses the [Inside Airbnb](http://insideairbnb.com/get-the-data/) Istanbul snapshot dated **September 29, 2025**. Four related files are downloaded during the data collection stage:

- `listings.csv.gz`
- `calendar.csv.gz`
- `reviews.csv.gz`
- `neighbourhoods.csv`

The analysis dataset is built at the listing level by combining cleaned listing information with calendar- and review-derived aggregates. The neighbourhood reference file was also audited during preprocessing; however, the listing file already contains the neighbourhood labels used in the final analysis dataset.


## Project Status

- ✅ Repository setup (March 17)
- ✅ Project proposal (March 31) - see [reports/proposal.md](reports/proposal.md)
- ✅ Data collection, EDA, and hypothesis tests (April 14)
- ⬜ Machine learning methods (May 5)
- ⬜ Final report and code submission (May 18)

## Repository Structure

```text
.
├── data/
│   └── README.md
├── figures/
├── notebooks/
│   └── 02_eda_hypothesis_tests.ipynb
├── reports/
│   ├── proposal.md
│   └── phase3_eda_hypothesis_tests.md
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   └── create_notebook.py
├── README.md
└── requirements.txt
```

## How to Reproduce

Raw and processed CSV files are excluded from version control by default to keep the repository lightweight. The analysis can be reproduced with the following steps:

```bash
pip install -r requirements.txt
python src/download_data.py
python src/preprocess.py
python src/create_notebook.py
jupyter notebook notebooks/02_eda_hypothesis_tests.ipynb
```

To verify that the notebook runs end-to-end:

```bash
jupyter nbconvert --to notebook --execute notebooks/02_eda_hypothesis_tests.ipynb --output /tmp/eda-check.ipynb
```

## Reports

- [Project proposal](reports/proposal.md)
- [Phase 3 milestone summary](reports/phase3_eda_hypothesis_tests.md)

## AI Usage

In accordance with the DSA 210 project guidelines, I explicitly declare that I used AI tools to help refine the project topic, improve repository organization, support the data analysis workflow, and assist with drafting and polishing documentation. All final project decisions, code review, and interpretation of results were completed by me.
