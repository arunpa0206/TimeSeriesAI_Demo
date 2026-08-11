"""Two-layer GCN on the Cora citation dataset.

Based on the GeeksforGeeks example:
https://www.geeksforgeeks.org/deep-learning/graph-neural-networks-with-pytorch/
"""

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


class CustomGNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = GCNConv(input_dim, hidden_dim)
        self.layer2 = GCNConv(hidden_dim, output_dim)

    def forward(self, feature_data: torch.Tensor, edge_info: torch.Tensor) -> torch.Tensor:
        x = self.layer1(feature_data, edge_info)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer2(x, edge_info)
        return F.log_softmax(x, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--data-root", default="data/Planetoid")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = Planetoid(root=args.data_root, name="Cora")
    graph_data = dataset[0]

    model = CustomGNN(dataset.num_node_features, 16, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(graph_data.x, graph_data.edge_index)
        loss = F.nll_loss(output[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"Epoch: {epoch + 1:03d}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(graph_data.x, graph_data.edge_index).argmax(dim=1)
    correct = (predictions[graph_data.test_mask] == graph_data.y[graph_data.test_mask]).sum()
    accuracy = int(correct) / int(graph_data.test_mask.sum())
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
