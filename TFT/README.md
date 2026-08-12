# Telecom Traffic Forecasting with TFT

This example uses Darts' Temporal Fusion Transformer (TFT) to forecast Internet
traffic for Telecom Italia's network area `SquareID=5060`.

The data is downloaded from the [O-RAN Software Community dataset repository](https://nexus3.o-ran-sc.org/repository/datasets/sms-call-internet-mi-2013-11-01_parsed.tar.gz)
on the first run. The script extracts the CSV from the tarball automatically.

## Run

From this `TFT` folder:

```bash
python -m pip install -r requirements.txt
python tft_telecom.py
```

For a quick smoke test, reduce training time:

```bash
python tft_telecom.py --epochs 1
```

Options include `--square-id`, `--epochs`, and `--data-dir`.
