# Graph Neural Network on Cora

This folder contains a two-layer Graph Convolutional Network (GCN) based on the
[GeeksforGeeks PyTorch GNN example](https://www.geeksforgeeks.org/deep-learning/graph-neural-networks-with-pytorch/).
It trains a node classifier on the Cora citation network using PyTorch Geometric.

## Run

From this `GNN` folder:

```bash
python -m pip install -r requirements.txt
python gnn_cora.py
```

The Cora dataset is downloaded automatically by PyTorch Geometric on the first run.
Use `python gnn_cora.py --epochs 20` for a quick test.
