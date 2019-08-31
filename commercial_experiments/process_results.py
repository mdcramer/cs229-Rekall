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

pre, rec, f1, tp, fp, fn = compute_prf1(preds, gt, 1)

print(
    'Seed: {}\t'
    'Pre: {pre:.4f}\t'
    'Rec: {rec:.4f}\t'
    'F1: {f1:.4f}'.format(
        seed, pre=pre, rec=rec, f1=f1
    )
)

print((pre, rec, f1, tp, fp, fn))
