import os
import sys
root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)
from evaluation import Evaluator
import numpy as np

labels = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0])
preds  = np.array([0,0,1,1,1,1,1,1,1,1,0,0,0])



print Evaluator('accuracy').compute_score(labels, preds)
print Evaluator('f1_score').compute_score(labels, preds)
print Evaluator('auc').compute_score(labels, preds)
print Evaluator('log-loss').compute_score(labels, preds)
print Evaluator('quadratic-loss').compute_score(labels, preds)

print Evaluator(['accuracy', 'f1_score', 'auc', 'log-loss', 'quadratic-loss']).compute_score(labels, preds)
print Evaluator(['accuracy']).compute_score(labels, preds)