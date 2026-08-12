"""Store-sales forecasting with Facebook Prophet.

Based on the GeeksforGeeks tutorial:
https://www.geeksforgeeks.org/data-analysis/time-series-forecasting-for-predicting-store-sales-using-prophet/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet


DATA_URL = "https://media.geeksforgeeks.org/wp-content/uploads/20240704211146/retail_sales_dataset.csv"


def load_data(csv_path: str | None = None) -> pd.DataFrame:
    """Load the tutorial's retail sales dataset from a local path or its URL."""
    return pd.read_csv(csv_path or DATA_URL)


def prepare_sales(data: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    """Aggregate sales by day and return Prophet's required ds/y columns."""
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    if category is not None:
        available = sorted(data["Product Category"].dropna().unique())
        data = data[data["Product Category"] == category]
        if data.empty:
            raise ValueError(f"Unknown category {category!r}. Available categories: {available}")

    daily_sales = data.groupby("Date", as_index=False)["Total Amount"].sum()
    return daily_sales.rename(columns={"Date": "ds", "Total Amount": "y"})


def forecast_sales(
    daily_sales: pd.DataFrame, periods: int, output_dir: Path, name: str
) -> pd.DataFrame:
    model = Prophet()
    model.fit(daily_sales)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    output_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_dir / f"{name}_forecast.csv", index=False)

    figure = model.plot(forecast)
    figure.suptitle(f"{name} Sales Forecast", y=1.02)
    figure.savefig(output_dir / f"{name}_forecast.png", bbox_inches="tight")
    plt.close(figure)

    components = model.plot_components(forecast)
    components.savefig(output_dir / f"{name}_components.png", bbox_inches="tight")
    plt.close(components)
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Optional local path to the retail sales CSV")
    parser.add_argument("--periods", type=int, default=365)
    parser.add_argument("--category", help="Forecast one category, e.g. Beauty")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    data = load_data(args.csv)
    daily_sales = prepare_sales(data, args.category)
    name = args.category.lower().replace(" ", "_") if args.category else "total_sales"
    forecast = forecast_sales(daily_sales, args.periods, args.output_dir, name)

    latest = forecast.iloc[-1]
    print(f"Forecast date: {latest['ds'].date()}")
    print(f"Predicted sales: {latest['yhat']:.2f}")
    print(f"Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
