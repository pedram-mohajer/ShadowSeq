import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("FILE.csv")

n_classes = 43
conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

# Group by sequence to treat each as one unit
sequence_groups = df.groupby("sequenceId")
for _, group in tqdm(sequence_groups, total=len(sequence_groups), desc="Processing sequences"):
    true_class = group['class_no_shadow'].iloc[0]
    pred_classes = group['class_shadow'].tolist()
    
    for pred in pred_classes:
        if pred != true_class:
            conf_matrix[true_class][pred] += 1


row_sums = conf_matrix.sum(axis=1, keepdims=True) + 1e-8 
conf_matrix_norm = conf_matrix / row_sums

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_norm, annot=False, cmap="Reds", square=True,
            xticklabels=np.arange(n_classes),
            yticklabels=np.arange(n_classes))
plt.xlabel("Predicted Class After Shadow")
plt.ylabel("True Class (Before Shadow)")
plt.title("Target Shift Distribution (Normalized)")
plt.tight_layout()


plt.savefig("target_shift_distribution_30.png", dpi=300)
print("Heatmap saved as target_shift_distribution.png")
