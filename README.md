# Lane 7 Demand Forecasting System

End-to-end demand forecasting platform built for a B2B apparel company to improve:

- Inventory planning
- Production planning
- Purchasing decisions
- SKU demand visibility
- Forecast scalability

---

# Business Problem

Lane 7 sells apparel products across:

- Styles
- Colors
- Sizes
- Product categories

The company needed a way to forecast future demand before orders arrive.

---

## Core Challenge

Direct SKU forecasting performed poorly because:

- Many SKU combinations had sparse historical demand
- Certain products had intermittent purchasing behavior
- Size/color combinations created fragmented demand patterns

Forecasting directly at the SKU level created unstable predictions.

---

# Solution Architecture

## 1) ETL Pipeline

Raw data was pulled from Lane 7's operational database:

- Orders
- OrderLine
- Item
- Size
- Color

The ETL pipeline cleans, joins, and transforms these datasets into forecasting-ready tables.

### ETL Outputs

### `dim_product`
Contains product-level metadata including:

- Style
- Color
- Size
- Product hierarchy details

### `gold_fact_monthly_demand.csv`

Primary training dataset used by the forecasting models.

Contains:

- Monthly product demand
- Historical sales volume
- Product relationships
- Time-based forecasting inputs

---

# Forecasting Pipeline

After ETL creates clean datasets, the forecasting pipeline:

- Segments products by demand behavior
- Engineers forecasting features
- Trains multiple forecasting models
- Selects best performing models
- Generates production forecasts

---

## Demand Segmentation

Products were segmented into:

- Regular demand
- Intermittent demand
- Low-volume demand

Each segment required different forecasting strategies.

---

# Models Used

Depending on segment and forecast horizon:

- XGBoost
- LightGBM
- Moving Average Models
- Seasonal Baselines

---

# SKU Allocation Layer

Instead of forecasting directly at SKU level:

1. Forecast demand at StyleCode level  
2. Allocate forecast down to SKU level  

This significantly improved forecast stability.

---

# Version Evolution

## V1

Initial forecasting baseline

---

## V2

Improved forecasting stability

Added:

- Better calibration
- Performance diagnostics
- Improved feature engineering

---

## V3

Final production winner

Added:

- Global bias control
- Improved production forecasting logic

---

## Comparison Framework 

- Cycling through the weights to find a better model than the current best
- Final validation framework used to compare all forecasting versions before deployment.

---

# Repository Structure

```bash
lane7-demand-forecasting/
│
├── etl_pipeline/
│   ├── v1/
│   └── v2/
│
├── forecasting_pipeline/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   └── comparison/
│
├── Notebooks/
│
└── README.md
```

---

# Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- Jupyter Notebook

---

# Business Value

This system helps Lane 7 improve:

- Inventory planning
- Production forecasting
- Purchasing strategy
- Demand visibility
- Operational scalability

---

# Future Improvements

- Size curve forecasting
- Hierarchical reconciliation
- Probabilistic forecasting
- Automated deployment pipeline

---

# Why This Project Matters

This project demonstrates:
- Data Engineering
- ETL Development
- Forecasting
- Machine Learning
- Feature Engineering
- Business Strategy
- Production-Oriented Data Science
