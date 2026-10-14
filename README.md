# Predictive Analytics Using Historical Data

## Thiranex – Task 3

### Objective
Build a predictive model to forecast future trends, preprocess historical data, evaluate model accuracy, and visualize predictions.

### Dataset
`historical_sales_raw.csv` is a **synthetic sample weekly sales dataset** created to demonstrate the workflow. If the official task tutorial requires a specific dataset, replace this sample with that dataset before final submission.

### Workflow
1. Load weekly historical sales.
2. Interpolate missing values.
3. Create week, month, quarter, lag, and rolling-average features.
4. Split chronologically into 80% training and 20% testing data.
5. Train a Random Forest Regressor.
6. Evaluate with MAE, RMSE, and R².
7. Forecast the next 12 weeks.
8. Save prediction tables and visualizations.

### Tools
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Jupyter Notebook.

### Evaluation
See `outputs/model_metrics.csv`. These metrics are calculated on the included synthetic sample data and are not real business results.

### Files
- `historical_sales_raw.csv` – sample input dataset
- `historical_sales_cleaned.csv` – cleaned dataset
- `predictive_analytics.py` – runnable Python script
- `predictive_analytics.ipynb` – notebook
- `outputs/` – metrics, predictions, and charts
- `requirements.txt` – required Python packages

### Run
Install dependencies using `pip install -r requirements.txt`, then run `python predictive_analytics.py`.

### Limitation
This is a demonstration project based on synthetic data. Use the official tutorial dataset if one is specified by Thiranex.
