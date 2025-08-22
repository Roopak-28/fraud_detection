import torch
from tgcn_model import TGCN
import argparse
import json
import os
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--metrics_out', type=str, required=True)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

data = torch.load(args.data_path)
with open(args.config_path) as f:
    config = json.load(f)
model = TGCN(config)
model.load_state_dict(torch.load(args.model_path))
model.eval()

with torch.no_grad():
    out = model.predict(data)
    pred = out.argmax(dim=1)
    test_pred = pred[data.test_mask].cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()
    known_mask = test_true != 2
    test_pred_known = test_pred[known_mask]
    test_true_known = test_true[known_mask]
    report = classification_report(test_true_known, test_pred_known, target_names=['Licit', 'Illicit'], digits=4, output_dict=True)
    print(report)

import json
with open(args.metrics_out, 'w') as f:
    json.dump(report, f)
print(f"Saved evaluation metrics to {args.metrics_out}")
