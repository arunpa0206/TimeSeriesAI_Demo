# Transformer-Based Telecom Traffic Forecasting

This example uses a small PyTorch Transformer encoder to predict the next
Internet-traffic reading from the previous 12 readings for Telecom Italia's
network area `SquareID=5060`.

The dataset is downloaded from the [O-RAN Software Community dataset repository](https://nexus3.o-ran-sc.org/repository/datasets/sms-call-internet-mi-2013-11-01_parsed.tar.gz)
on the first run. The script extracts the CSV from the tarball automatically.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python transformer_forecasting.py
```

For a quick smoke test:

```bash
python transformer_forecasting.py --epochs 1
```

Options include `--square-id`, `--epochs`, and `--data-dir`.
