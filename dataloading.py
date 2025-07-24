import torch
from torch_geometric.datasets import EllipticBitcoinDataset
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

os.makedirs(args.root_path, exist_ok=True)
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

print("Loading Elliptic Bitcoin dataset...")
dataset = EllipticBitcoinDataset(root=args.root_path)
data = dataset[0]
torch.save(data, args.output_path)
print(f"Saved raw data to {args.output_path}")
