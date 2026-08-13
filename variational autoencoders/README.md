# Telecom Traffic Anomaly Detection with a Variational Autoencoder

This example uses a variational autoencoder with a `12 → 8 → 4` encoder and a
`4 → 8 → 12` decoder. It learns a normal baseline from Telecom Italia traffic
windows and flags the latest window when its reconstruction error exceeds the
95th-percentile training threshold.

The dataset is downloaded and extracted automatically on the first run.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python vae_telecom.py
```

For a quick smoke test:

```bash
python vae_telecom.py --epochs 2
```
