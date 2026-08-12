# Store Sales Forecasting with Prophet

This folder implements the main example from the [GeeksforGeeks Prophet tutorial](https://www.geeksforgeeks.org/data-analysis/time-series-forecasting-for-predicting-store-sales-using-prophet/).
It downloads the tutorial's retail-sales CSV, aggregates `Total Amount` by day,
fits a Prophet model, forecasts 365 days, and saves the forecast and component
plots under `outputs/`.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python prophet_store_sales.py
```

Forecast a single product category:

```bash
python prophet_store_sales.py --category Beauty
```

Use a local CSV or change the forecast horizon:

```bash
python prophet_store_sales.py --csv path/to/retail_sales_dataset.csv --periods 30
```
