

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

The cleaning stage handles missing data explicitly: rows are removed when the target `price` is missing or unusable, extremely sparse columns (>80% missing) are dropped, and meaningful review-related missingness is preserved rather than globally imputed away.

## Phase 3 Outputs

The April 14 milestone is completed in this repository through the following files:

- [src/download_data.py](src/download_data.py): downloads the raw Istanbul Inside Airbnb files
- [src/preprocess.py](src/preprocess.py): cleans, filters, and merges the data into a listing-level dataset
- [src/create_notebook.py](src/create_notebook.py): generates the analysis notebook
- [notebooks/02_eda_hypothesis_tests.ipynb](notebooks/02_eda_hypothesis_tests.ipynb): full EDA and hypothesis testing notebook
- [reports/phase3_eda_hypothesis_tests.md](reports/phase3_eda_hypothesis_tests.md): written Phase 3 milestone summary
- [figures/](figures): exported visualizations from the notebook

## Phase 4 Outputs

The May 5 machine learning milestone is implemented through the following files:

- [src/modeling_utils.py](src/modeling_utils.py): shared helpers for modeling data preparation and metrics
- [src/train_price_models.py](src/train_price_models.py): trains and compares multiple regression models
- [src/create_modeling_notebook.py](src/create_modeling_notebook.py): generates the modeling notebook
- [notebooks/03_modeling.ipynb](notebooks/03_modeling.ipynb): executed machine learning notebook for price prediction, including saved outputs
- [reports/phase4_machine_learning.md](reports/phase4_machine_learning.md): written Phase 4 milestone summary with model-comparison visuals

## Final Outputs

The final submission package includes:

- [reports/final_report.md](reports/final_report.md): final project report combining motivation, data source, EDA, hypothesis tests, machine learning, findings, limitations, and future work
- executed notebooks with saved outputs:
  - [notebooks/02_eda_hypothesis_tests.ipynb](notebooks/02_eda_hypothesis_tests.ipynb)
  - [notebooks/03_modeling.ipynb](notebooks/03_modeling.ipynb)
- [figures/](figures): EDA, hypothesis-test, and machine-learning visualizations used in the reports

## Key Findings So Far

- The final analysis dataset contains **25,206 listings** and **81 features** after cleaning.
- Listing price is strongly right-skewed. The **median price is 2,535 TRY** and the **mean price is 3,691 TRY**.
- **Entire home/apt** listings make up about **71.6%** of the dataset and have the highest median price (**2,921 TRY**).
- **Superhost** listings have a higher median price (**3,380 TRY**) than non-superhost listings (**2,355 TRY**).
- Review score rating is positively associated with price, but the relationship is **weak** (`Spearman rho = 0.148`).
- Price differs significantly across both **room types** and **major neighbourhood groups**.
- Room type and availability level are significantly associated, but the effect size is **small**.
- For the machine learning stage, **Random Forest** performed best for price prediction with **test RMSE = 3,910 TRY**, **test MAE = 1,328 TRY**, and **R² = 0.287**.
- The strongest predictive signals in the best model include **accommodates, property type, longitude, bathrooms, latitude, and host response rate**.

## Selected Visuals

### Exploratory Data Analysis

![Price Distribution](figures/01_price_distribution.png)

![Price by Room Type](figures/06_price_by_room_type.png)

![Correlation Heatmap](figures/10_correlation_heatmap.png)

### Machine Learning

![Model Comparison](figures/15_model_comparison.png)

![Actual vs Predicted Prices](figures/16_actual_vs_predicted.png)

![Random Forest Feature Importance](figures/18_random_forest_feature_importance.png)

## Project Status

- ✅ Repository setup (March 17)
- ✅ Project proposal (March 31) - see [reports/proposal.md](reports/proposal.md)
- ✅ Data collection, EDA, and hypothesis tests (April 14)
- ✅ Machine learning methods (May 5)
- ✅ Executed notebooks with saved outputs and visual report updates
- ✅ Final report and code submission package (May 18)

## Repository Structure

```text
.
├── data/
│   └── README.md
├── figures/
├── notebooks/
│   └── 02_eda_hypothesis_tests.ipynb
│   └── 03_modeling.ipynb
├── reports/
│   ├── proposal.md
│   └── phase3_eda_hypothesis_tests.md
│   └── phase4_machine_learning.md
│   └── final_report.md
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   └── create_notebook.py
│   ├── modeling_utils.py
│   ├── train_price_models.py
│   └── create_modeling_notebook.py
├── AI_USAGE.md
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
python src/train_price_models.py
python src/create_modeling_notebook.py
jupyter nbconvert --to notebook --execute notebooks/02_eda_hypothesis_tests.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/03_modeling.ipynb --inplace
```

To verify that the notebook runs end-to-end:

```bash
jupyter nbconvert --to notebook --execute notebooks/02_eda_hypothesis_tests.ipynb --output /tmp/eda-check.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_modeling.ipynb --output /tmp/model-check.ipynb
```

## Reports

- [Project proposal](reports/proposal.md)
- [Phase 3 milestone summary](reports/phase3_eda_hypothesis_tests.md)
- [Phase 4 machine learning summary](reports/phase4_machine_learning.md)
- [Final project report](reports/final_report.md)
- [AI usage disclosure](AI_USAGE.md)

## AI Usage

In accordance with the DSA 210 project guidelines, I explicitly declare that I used AI tools to help refine the project topic, improve repository organization, support the data analysis workflow, and assist with drafting and polishing documentation. All final project decisions, code review, and interpretation of results were completed by me.



