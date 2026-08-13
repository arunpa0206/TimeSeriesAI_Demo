"""Detect unusual Telecom Italia traffic with a variational autoencoder."""

import argparse
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


DATA_URL = (
    "https://nexus3.o-ran-sc.org/repository/datasets/"
    "sms-call-internet-mi-2013-11-01_parsed.tar.gz"
)
WINDOW_SIZE = 12


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
            raise FileNotFoundError("No CSV file was found inside the archive.")
        csv_path = data_dir / Path(csv_member.name).name
        if not csv_path.exists():
            csv_path.write_bytes(archive.extractfile(csv_member).read())

    df = pd.read_csv(csv_path)
    required = {"SquareID", "Timestamp", "InternetTraffic"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    traffic = (
        df[df["SquareID"] == square_id]
        .sort_values("Timestamp")["InternetTraffic"]
        .to_numpy(dtype=np.float32)
    )
    if traffic.size <= WINDOW_SIZE:
        raise ValueError(f"Not enough readings found for SquareID {square_id}.")
    return traffic


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(WINDOW_SIZE, 8), nn.ReLU())
        self.mean = nn.Linear(8, 4)
        self.log_variance = nn.Linear(8, 4)
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, WINDOW_SIZE), nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.mean(encoded), self.log_variance(encoded)

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * log_variance)
        random_noise = torch.randn_like(standard_deviation)
        return mean + random_noise * standard_deviation

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(x)
        latent = self.reparameterize(mean, log_variance)
        return self.decoder(latent), mean, log_variance


def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    reconstruction_loss = F.mse_loss(reconstructed, original, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    return (reconstruction_loss + kl_loss) / original.size(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--square-id", type=int, default=5060)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    torch.manual_seed(42)
    traffic = load_traffic(args.data_dir, args.square_id)
    traffic_min, traffic_max = traffic.min(), traffic.max()
    if traffic_max == traffic_min:
        raise ValueError("Traffic values have zero range; normalization is undefined.")
    normalized = (traffic - traffic_min) / (traffic_max - traffic_min)
    windows = np.asarray(
        [normalized[i : i + WINDOW_SIZE] for i in range(len(normalized) - WINDOW_SIZE + 1)],
        dtype=np.float32,
    )

    split = int(len(windows) * 0.8)
    train = torch.tensor(windows[:split])
    test = torch.tensor(windows[split:])
    loader = DataLoader(TensorDataset(train), batch_size=16, shuffle=True)

    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        model.train()
        for (batch,) in loader:
            reconstructed, mean, log_variance = model(batch)
            loss = vae_loss(reconstructed, batch, mean, log_variance)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch + 1:03d}, Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        train_mean, _ = model.encode(train)
        train_reconstructed = model.decoder(train_mean)
        train_error = ((train_reconstructed - train) ** 2).mean(dim=1)
        threshold = torch.quantile(train_error, 0.95)

        latest = test[-1:]
        latest_mean, _ = model.encode(latest)
        latest_reconstructed = model.decoder(latest_mean)
        latest_error = ((latest_reconstructed - latest) ** 2).mean()

    print("Reconstruction error:", round(latest_error.item(), 6))
    print("Anomaly threshold:", round(threshold.item(), 6))
    if latest_error > threshold:
        print("Maintenance alert: Unusual network traffic detected.")
    else:
        print("Network traffic is normal.")


if __name__ == "__main__":
    main()
