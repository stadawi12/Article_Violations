import pickle
import numpy as np

import torch

from torchmetrics.functional import accuracy, auroc

from sklearn.metrics import classification_report

import pandas as pd

THRESHOLD = 0.5
LABEL_COLUMNS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

with open('val_preds_labels.pickle', 'rb') as v_pkl:
    val_preds, val_labels = pickle.load(v_pkl)

with open('test_preds_labels.pickle', 'rb') as t_pkl:
    test_preds, test_labels = pickle.load(t_pkl)

val_acc = accuracy(val_preds, val_labels, threshold=THRESHOLD)
test_acc = accuracy(test_preds, test_labels, threshold=THRESHOLD)

print("total accuracy:", val_acc, "- validation")
print("total accuracy:", test_acc,"- test\n")

val_count = 0
for i in range(1000):
    val_p = [0 if i <= 0.5 else 1 for i in val_preds[i]]
    if torch.equal(torch.tensor(val_p, dtype=torch.int32), val_labels[i]):
        val_count += 1
test_count = 0
for i in range(1000):
    test_p = [0 if i <= 0.5 else 1 for i in test_preds[i]]
    if torch.equal(torch.tensor(test_p, dtype=torch.int32), 
            test_labels[i]):
        test_count += 1

print("number of totally correct predictions:", val_count, "/ 1000 - validation")
print("number of totally correct predictions:", test_count, "/ 1000 - test\n")

print("AUROC per article")
for i, name in enumerate(LABEL_COLUMNS):
    val_auroc = auroc(val_preds[:, i], val_labels[:, i], pos_label=1)
    test_auroc = auroc(test_preds[:, i], test_labels[:, i], pos_label=1)
    print(f"{name}: {val_auroc} - validation")
    print(f"{name}: {test_auroc} - test\n")

y_pred = test_preds.numpy()
y_true = test_labels.numpy()

upper, lower = 1, 0

y_pred = np.where(y_pred > THRESHOLD, upper, lower)

report = classification_report(
    y_true,
    y_pred,
    target_names=LABEL_COLUMNS,
    zero_division=0,
    output_dict=True
    )

df = pd.DataFrame(report).transpose()

print(df)

df.to_csv('docs/test_report.csv')

print(auroc(val_preds, val_labels, num_classes=10, average='weighted'))
