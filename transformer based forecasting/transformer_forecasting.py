"""Forecast Telecom Italia traffic with a small PyTorch Transformer encoder.

The example uses the O-RAN Software Community Telecom Italia dataset and the
previous 12 readings to predict the next reading.
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


DATA_URL = (
    "https://nexus3.o-ran-sc.org/repository/datasets/"
    "sms-call-internet-mi-2013-11-01_parsed.tar.gz"
)
LOOKBACK = 12


def load_traffic(data_dir: Path, square_id: int) -> np.ndarray:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "telecom.tar.gz"

    if not archive_path.exists():
        print(f"Downloading dataset to {archive_path} ...")
        urllib.request.urlretrieve(DATA_URL, archive_path)

    with tarfile.open(archive_path, mode="r:gz") as archive:
        csv_member = next(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith(".csv")
            ),
            None,
        )
        if csv_member is None:
            raise FileNotFoundError("No CSV file was found inside the downloaded archive.")
        csv_path = data_dir / Path(csv_member.name).name
        if not csv_path.exists():
            csv_path.write_bytes(archive.extractfile(csv_member).read())

    df = pd.read_csv(csv_path)
    required = {"SquareID", "timestamp", "InternetTraffic"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    traffic = (
        df[df["SquareID"] == square_id]
        .groupby("timestamp")["InternetTraffic"]
        .sum()
        .to_numpy(dtype=np.float32)
    )
    if traffic.size <= LOOKBACK:
        raise ValueError(f"Not enough readings found for SquareID {square_id}.")
    return traffic


class TrafficTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_layer = nn.Linear(16 * LOOKBACK, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input_projection(values)
        values = self.encoder(values)
        return self.output_layer(values.flatten(start_dim=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--square-id", type=int, default=5060)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    torch.manual_seed(42)
    traffic = load_traffic(args.data_dir, args.square_id)
    mean, std = traffic.mean(), traffic.std()
    if std == 0:
        raise ValueError("Traffic values have zero variance; normalization is undefined.")
    data = (traffic - mean) / std

    inputs, targets = [], []
    for index in range(len(data) - LOOKBACK):
        inputs.append(data[index : index + LOOKBACK])
        targets.append(data[index + LOOKBACK])

    x = torch.tensor(np.asarray(inputs), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32).unsqueeze(-1)

    model = TrafficTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        prediction = model(x)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"Epoch: {epoch + 1:03d}, Loss: {loss.item():.6f}")

    latest = torch.tensor(data[-LOOKBACK:], dtype=torch.float32).reshape(1, LOOKBACK, 1)
    model.eval()
    with torch.no_grad():
        forecast = model(latest).item() * std + mean

    print("Predicted network traffic:", round(float(forecast), 2))
    if forecast > np.percentile(traffic, 90):
        print("Maintenance alert: Check tower capacity, power and cooling.")
    else:
        print("Network operating normally.")


if __name__ == "__main__":
    main()
