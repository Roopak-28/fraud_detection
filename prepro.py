import torch
import argparse
import os

def analyze_class_distribution(data):
    train_labels = data.y[data.train_mask]
    known_mask = train_labels != 2
    known_labels = train_labels[known_mask]
    licit_count = (known_labels == 0).sum().item()
    illicit_count = (known_labels == 1).sum().item()
    total_known = len(known_labels)
    ce_weight_licit = total_known / (2 * licit_count) if licit_count > 0 else 1
    ce_weight_illicit = total_known / (2 * illicit_count) if illicit_count > 0 else 1
    class_weights = [ce_weight_licit, ce_weight_illicit, 0.0]
    alpha_licit = illicit_count / total_known
    alpha_illicit = licit_count / total_known
    alpha_weights = [alpha_licit, alpha_illicit, 0.0]
    return alpha_weights, class_weights

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--data_out', type=str, required=True)
parser.add_argument('--weights_out', type=str, required=True)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.data_out), exist_ok=True)
os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)

data = torch.load(args.input_path)
alpha_weights, class_weights = analyze_class_distribution(data)

# Save processed data (can add normalization/encoding here)
torch.save(data, args.data_out)

# Save weights as a dict
import json
weights_dict = {'focal_loss_alpha': alpha_weights, 'class_weights': class_weights}
with open(args.weights_out, 'w') as f:
    json.dump(weights_dict, f)

print(f"Saved processed data to {args.data_out}")
print(f"Saved weights to {args.weights_out}")
