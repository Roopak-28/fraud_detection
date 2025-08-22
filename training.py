import torch
from tgcn_model import TGCN
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--model_out', type=str, required=True)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

data = torch.load(args.data_path)
with open(args.weights_path) as f:
    weights = json.load(f)
# You can load more config as needed
# Example config:
config = {
    'input_dim': data.x.shape[1],
    'output_dim': int(data.y.max()) + 1,
    'hidden_dim': 32,
    'epochs': 50,
    'focal_loss_alpha': weights['focal_loss_alpha'],
    'class_weights': weights['class_weights'],
}
model = TGCN(config)

print("Starting training...")
for epoch in range(config['epochs']):
    loss = model.train_step(data, data.train_mask)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

torch.save(model.state_dict(), args.model_out)
print(f"Saved trained model to {args.model_out}")
