import json
import pickle
import numpy as np
import sys

seed = sys.argv[1]

with open('results_dense_{}/test.json'.format(seed), 'r') as f:
    results = json.load(f)

gt = [
    res[0]['gt']
    for res in results['results']
]

preds = [
    res[np.argmax([pred['score'] for pred in res])]['label']
    for res in results['results']
]

def compute_prf1(preds, gts, class_name):
    tp = len([1 for pred, gt in zip(preds, gts) if gt == class_name and pred == gt])
    fp = len([1 for pred, gt in zip(preds, gts) if pred == class_name and pred != gt])
    fn = len([1 for pred, gt in zip(preds, gts) if gt == class_name and pred != gt])

    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * pre * rec  / (pre + rec)

    return pre, rec, f1, tp, fp, fn

results = []
for class_name in ['long', 'medium', 'close_up']:
    results.append((class_name, compute_prf1(preds, gt, class_name)))

avg_pre = np.mean([res[1][0] for res in results])
avg_rec = np.mean([res[1][1] for res in results])
avg_f1 = np.mean([res[1][2] for res in results])

print(
    'Seed: {}\t'
    'Pre: {pre:.4f}\t'
    'Rec: {rec:.4f}\t'
    'F1: {f1:.4f}'.format(
        seed, pre=avg_pre, rec=avg_rec, f1=avg_f1
    )
)

for class_name, res in results:
    print(class_name, res)
