import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cm = np.array([
    [369, 1, 23],
    [5, 268, 28],
    [178, 47, 0],
])

labels = ["human", "dummy", "background"]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="g",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True
)

plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("runs/thermal_v1/train_v1_360/confusion_matrix_human_dummy.png", dpi=300)
plt.close()
