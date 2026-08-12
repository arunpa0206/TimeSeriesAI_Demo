"""Forecast Telecom Italia network traffic with a Darts TFT model.

The source dataset is published by the O-RAN Software Community:
https://nexus3.o-ran-sc.org/repository/datasets/sms-call-internet-mi-2013-11-01_parsed.tar.gz
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel


DATA_URL = (
    "https://nexus3.o-ran-sc.org/repository/datasets/"
    "sms-call-internet-mi-2013-11-01_parsed.tar.gz"
)


def load_traffic(data_dir: Path, square_id: int) -> np.ndarray:
    """Download/extract the archive and return traffic for one network area."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "telecom.tar.gz"

    if not archive_path.exists():
        print(f"Downloading dataset to {archive_path} ...")
        urllib.request.urlretrieve(DATA_URL, archive_path)

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        csv_member = next(
            (member for member in members if member.name.lower().endswith(".csv")),
            None,
        )
        if csv_member is None:
            raise FileNotFoundError("No CSV file was found inside the downloaded archive.")
        extracted_csv = data_dir / Path(csv_member.name).name
        if not extracted_csv.exists():
            extracted_csv.write_bytes(archive.extractfile(csv_member).read())

    df = pd.read_csv(extracted_csv)
    required = {"SquareID", "timestamp", "InternetTraffic"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    traffic = (
        df[df["SquareID"] == square_id]
        .groupby("timestamp")["InternetTraffic"]
        .sum()
        .to_numpy(dtype="float32")
    )
    if traffic.size < 20:
        raise ValueError(f"Only {traffic.size} traffic values found for SquareID {square_id}.")
    return traffic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--square-id", type=int, default=5060)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    traffic = load_traffic(args.data_dir, args.square_id)
    series = TimeSeries.from_values(traffic)

    model = TFTModel(
        input_chunk_length=12,
        output_chunk_length=1,
        hidden_size=8,
        lstm_layers=1,
        num_attention_heads=2,
        n_epochs=args.epochs,
        add_relative_index=True,
        random_state=42,
    )
    model.fit(series)

    forecast = float(model.predict(1).values()[0][0])
    print("Predicted network traffic:", round(forecast, 2))

    if forecast > np.percentile(traffic, 90):
        print("Maintenance alert: Check tower capacity, power and cooling.")
    else:
        print("Network operating normally.")


if __name__ == "__main__":
    main()
