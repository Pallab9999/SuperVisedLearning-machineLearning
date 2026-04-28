# ============================================================
# Classifier Comparison: 5x2 Cross-Validation + Friedman Test
# Classifiers: SVM Linear, SVM RBF, KNN
# Significance level: alpha = 0.05
# ============================================================

# -------------------------------------------------------
# CELL 1 — Imports
# -------------------------------------------------------
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from scipy.stats import friedmanchisquare

np.random.seed(42)  # reproducibility


# -------------------------------------------------------
# CELL 2 — 5x2 Cross-Validation Loop
# -------------------------------------------------------

classifiers = {
    # Linear SVM baseline (good when classes are linearly separable)
    'SVM Linear': SVC(kernel='linear', C=1.0),
    # Non‑linear SVM with RBF kernel (can model curved decision boundaries)
    'SVM RBF':    SVC(kernel='rbf', C=1.0),
    # k‑Nearest Neighbours with Euclidean distance and k=5
    'KNN':        KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
}

# Dictionary that will store, for each classifier, one mean 5x2‑CV accuracy
# per dataset: clf_name -> [acc_ds1, acc_ds2, acc_ds3, acc_ds4]
all_results = {}

for clf_name, clf_template in classifiers.items():

    # This list will contain the 5x2‑CV accuracy for the 4 datasets
    accuracy_5x2 = []

    for ndataset in range(1, 5):

        # Load the ndataset‑th .mat file (features in 'data', labels in 'labels')
        dataset = sio.loadmat(f'dataset{ndataset}.mat')
        data   = dataset['data']              # shape: (N, 2)
        labels = dataset['labels'].flatten()  # shape: (N,), values: 1 or 2

        # Will store the 5 accuracy values coming from the 5 repetitions
        accuracy_times = []

        for ntimes in range(5):

            idx_tr, idx_te = [], []

            # Stratified split: for each class, use half of the samples for
            # training and the other half for testing
            for nclass in [1, 2]:
                class_indices = np.where(labels == nclass)[0]
                np.random.shuffle(class_indices)
                half = len(class_indices) // 2
                idx_tr.extend(class_indices[:half])   # first half → train
                idx_te.extend(class_indices[half:])   # second half → test

            labels_tr, labels_te = labels[idx_tr], labels[idx_te]
            data_tr,   data_te   = data[idx_tr, :],  data[idx_te, :]

            # First direction of the 2‑fold CV: train on train split, test on test
            clf = clone(clf_template)  # fresh, unfitted copy of the classifier
            clf.fit(data_tr, labels_tr)
            acc1 = np.sum(clf.predict(data_te) == labels_te) / len(labels_te)

            # Second direction: swap the roles (train on former test, test on train)
            clf.fit(data_te, labels_te)
            acc2 = np.sum(clf.predict(data_tr) == labels_tr) / len(labels_tr)

            # Accuracy for this repetition = mean of the two directions
            accuracy_times.append((acc1 + acc2) / 2)

        # 5x2‑CV accuracy for this dataset = mean over the 5 repetitions
        accuracy_5x2.append(np.mean(accuracy_times))

    # Store the 4 accuracies (one per dataset) for this classifier
    all_results[clf_name] = accuracy_5x2

# --- Print accuracy table ---
print("=" * 66)
print("5x2 CV Accuracies")
print("=" * 66)
print(f"{'Classifier':<14} | {'DS1':>6} | {'DS2':>6} | {'DS3':>6} | {'DS4':>6} | {'Avg':>6}")
print("-" * 66)
for name, accs in all_results.items():
    avg_acc = np.mean(accs)
    vals = " | ".join(f"{a:>6.4f}" for a in accs)
    print(f"{name:<14} | {vals} | {avg_acc:>6.4f}")


# -------------------------------------------------------
# CELL 3 — Friedman Test + Average Ranks
# -------------------------------------------------------

k = 3  # number of classifiers
N = 4  # number of datasets

# Build accuracy matrix: each row = dataset, each column = classifier
acc_matrix = np.array([all_results[n] for n in classifiers]).T  # shape (N, k)

# Rank classifiers per dataset (row‑wise).
# We sort accuracies in descending order so that rank 1 = best accuracy.
ranks = np.zeros_like(acc_matrix)
for i in range(N):
    order = np.argsort(-acc_matrix[i])        # descending
    for rank_pos, col in enumerate(order):
        ranks[i, col] = rank_pos + 1

avg_ranks = ranks.mean(axis=0)  # one average rank per classifier

print("\n" + "=" * 40)
print("Ranks Matrix (1 = best per dataset)")
print("=" * 40)
print(f"{'':14} | {'DS1':>4} | {'DS2':>4} | {'DS3':>4} | {'DS4':>4} | {'Avg':>5}")
print("-" * 44)
for name, row, avg in zip(classifiers.keys(), ranks, avg_ranks):
    print(f"{name:<14} | " + " | ".join(f"{int(r):>4}" for r in row) + f" | {avg:>5.2f}")

# Friedman test on the rank matrix (non‑parametric equivalent of repeated‑measures ANOVA)
stat, p_value = friedmanchisquare(*[ranks[:, j] for j in range(k)])
print(f"\nFriedman statistic : {stat:.4f}")
print(f"p-value            : {p_value:.4f}")
print(f"Significant (p<0.05): {p_value < 0.05}")

# Critical Difference (Nemenyi post‑hoc)
# CD = q_alpha * sqrt(k*(k+1) / (6*N))
# For k=3 classifiers and alpha=0.05, q_alpha = 2.343 (from Nemenyi tables)
q_alpha = 2.343
CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
print(f"Critical Difference: {CD:.4f}  (alpha=0.05, k={k}, N={N})")


# -------------------------------------------------------
# CELL 4 — Figure 1: Dataset Scatter Plots
# -------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.patch.set_facecolor('white')

for ds_idx in range(4):
    ax = axes[ds_idx]
    # Reload each dataset to plot its 2D scatter (features vs. labels)
    dataset = sio.loadmat(f'dataset{ds_idx+1}.mat')
    data   = dataset['data']
    labels = dataset['labels'].flatten()

    for cls, color, marker, lbl in zip([1, 2], ['#4C72B0', '#DD8452'], ['o', 's'], ['Class 1', 'Class 2']):
        mask = labels == cls
        ax.scatter(data[mask, 0], data[mask, 1], c=color, marker=marker, s=18, alpha=0.6, label=lbl)

    n  = len(labels)
    c1 = int(np.sum(labels == 1))
    c2 = int(np.sum(labels == 2))
    ax.set_title(f'Dataset {ds_idx+1}\n(N={n}, C1={c1}, C2={c2})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlabel('Feature 1', fontsize=9)
    ax.set_ylabel('Feature 2', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')

plt.suptitle('Dataset Visualizations', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_datasets.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# -------------------------------------------------------
# CELL 5 — Figure 2: CD Diagram + Accuracy Bar Chart
# -------------------------------------------------------

clf_names      = list(classifiers.keys())
avg_ranks_list = avg_ranks.tolist()
colors         = ['#4C72B0', '#DD8452', '#55A868']

# Figure 2:  The Critical Difference diagram.
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
fig.patch.set_facecolor('white')

# --- Critical Difference Diagram ---
ax.set_facecolor('white')
y_positions = [3, 2, 1]  # one row per classifier

# CD bracket at top
top_y = 3.7
ax.annotate('', xy=(avg_ranks_list[1] + CD/2, top_y),
            xytext=(avg_ranks_list[1] - CD/2, top_y),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(avg_ranks_list[1], top_y + 0.13, f'CD = {CD:.2f}',
        ha='center', va='bottom', fontsize=10, fontweight='bold')

# Classifier bars
for name, rank, ypos, color in zip(clf_names, avg_ranks_list, y_positions, colors):
    ax.plot([rank - CD/2, rank + CD/2], [ypos, ypos], color=color, lw=4, solid_capstyle='round')
    ax.plot(rank, ypos, 'o', color=color, markersize=10, zorder=5)
    ax.text(-0.1, ypos, name, ha='right', va='center', fontsize=11, fontweight='bold', color=color)
    ax.text(rank, ypos - 0.28, f'avg rank = {rank:.1f}',
            ha='center', va='top', fontsize=8.5, color='#555')

# Dashed vertical lines at integer ranks
for r in [1, 2, 3]:
    ax.axvline(x=r, color='gray', linestyle='--', lw=0.9, alpha=0.5)

ax.set_xlim(0.2, 4.2)
ax.set_ylim(0.2, 4.3)
ax.set_xlabel('Average Rank  (1 = best)', fontsize=11)
ax.set_xticks([1, 2, 3])
ax.set_yticks([])
ax.set_title('Critical Difference Diagram\n(Nemenyi post-hoc, \u03b1 = 0.05)', fontsize=12, fontweight='bold')
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

sig_text = (f"p = {p_value:.4f} > 0.05  \u2192  Not significant\n"
            "All bars overlap: no classifier\nis significantly better or worse")
ax.text(-0.05, 0.98, sig_text, transform=ax.transAxes,
        ha='right', va='top', fontsize=9.5, clip_on=False,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3CD', edgecolor='#FFC107', alpha=0.95))

plt.suptitle('Classifier Evaluation & Comparison \u2014 Friedman Test',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_results.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Done!")
