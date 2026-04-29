# Lane 7 Demand Forecasting System

End-to-end demand forecasting pipeline built for a B2B apparel company to improve inventory planning, purchasing decisions, production forecasting, and SKU allocation.

---

## Business Problem

Lane 7 sells apparel products across:

- Multiple styles
- Multiple colors
- Multiple sizes
- Wholesale-driven demand cycles

The biggest challenge was forecasting demand at the SKU level, where demand was often sparse and inconsistent.

### Initial Problem:
Direct SKU forecasting produced weak results because:

- Many SKUs had low historical volume
- Demand was highly fragmented across size/color combinations
- Certain products had intermittent ordering behavior

---

## Solution Architecture

Instead of forecasting directly at the SKU level:

### Step 1: Forecast at StyleCode Level
This created more stable demand signals.

### Step 2: Segment Products Based on Demand Behavior

Products were segmented into:

- Regular demand
- Intermittent demand
- Low-volume demand

---

## Modeling Approach

Different forecasting models were tested depending on demand behavior:

- XGBoost
- LightGBM
- Moving Average Models
- Seasonal Baselines
- Allocation Optimization Models

The pipeline dynamically selected the best-performing model based on product segment and forecast horizon.

---

## Version Evolution

### V1 (v7.4)
Initial production forecasting pipeline

### V2 (v7.5)
Improved forecasting stability and feature engineering

### V3 (v7.6)
Final production winner

### V8
Comparison framework used to validate model performance and deployment readiness

---

## Project Structure

```bash
forecasting_pipeline/
│
├── data_prep.py
├── features.py
├── segmentation.py
├── models.py
├── evaluation.py
├── allocation.py
├── holdout_v6.py
├── production_outputs_v76.py
```

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- Jupyter Notebook

---

## Business Impact

This project helps businesses improve:

- Inventory planning
- Purchasing decisions
- Production planning
- Demand visibility
- Forecast scalability

---

## Future Improvements

- Size curve forecasting
- Hierarchical reconciliation
- Probabilistic forecasting
- Automated deployment pipeline

---

## Why This Project Matters

This project demonstrates:
- Forecasting  
- Machine Learning  
- Feature Engineering  
- Data Engineering  
- Business Strategy  
- Production-Oriented Thinking
