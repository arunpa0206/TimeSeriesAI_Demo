# Telecom Traffic Anomaly Detection with an Autoencoder

This example uses a dense autoencoder with a `12 → 8 → 4` encoder and a
`4 → 8 → 12` decoder to detect unusual Telecom Italia network traffic.

The Telecom Italia dataset is downloaded and extracted automatically on the
first run. The first 80% of traffic windows train the normal baseline. A window
is flagged when its reconstruction error exceeds the 95th-percentile training
error threshold.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python autoencoder_telecom.py
```

For a quick smoke test:

```bash
python autoencoder_telecom.py --epochs 2
```
