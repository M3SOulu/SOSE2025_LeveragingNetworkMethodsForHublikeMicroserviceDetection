import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import scikit_posthocs as sp
import os

# === Create output directory ===
output_dir = "RANKSResults"
os.makedirs(output_dir, exist_ok=True)

# === Load CSV ===
df = pd.read_csv("Metrics/metrics_centrality.csv")

# === Select centrality columns ===
centrality_cols = [
    'Norm. Degree Centrality', #'Degree Centrality',
    'Norm. In-degree Centrality', #'In-degree Centrality',
    'Norm. Out-degree Centrality', #'Out-degree Centrality',
    'Eigenvector Centrality', 'Betweenness Centrality',
    'Closeness Centrality', 'PageRank Centrality',
    'Hub Score', 'Authority Score',
    'Subgraph Centrality', 'Clustering Coefficient'
]
centrality_df = df[centrality_cols]

# === Rank data for each centrality ===
ranked_df = centrality_df.rank(axis=0, method='average')

# === 1. Spearman Correlation ===
spearman_corr = ranked_df.corr(method='spearman')
spearman_corr.to_csv(os.path.join(output_dir, "spearman_correlation_matrix.csv"))

# === Plot Spearman heatmap ===
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", square=True,
            cbar_kws={'label': 'Spearman Rank Correlation'})
plt.title("Heatmap of Spearman Rank Correlation Between Centrality Measures")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spearman_correlation_heatmap.png"))
plt.close()

# === Dendrogram based on 1 - Spearman ===
dist_matrix = 1 - spearman_corr
condensed_dist = squareform(dist_matrix.values)
linkage_matrix = linkage(condensed_dist, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=spearman_corr.columns, leaf_rotation=90)
plt.title("Dendrogram of Centrality Metric Similarities (1 - Spearman)")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "centrality_dendrogram.png"))
plt.close()

# === 2. Friedman Test ===
friedman_stat, friedman_p = friedmanchisquare(*[ranked_df[col] for col in ranked_df.columns])
with open(os.path.join(output_dir, "friedman_test_result.txt"), "w") as f:
    f.write(f"Friedman Test Statistic: {friedman_stat:.4f}\n")
    f.write(f"P-value: {friedman_p:.10f}\n")

# === 3. Nemenyi Post-hoc Test ===
nemenyi_result = sp.posthoc_nemenyi_friedman(ranked_df)
nemenyi_result.to_csv(os.path.join(output_dir, "nemenyi_posthoc_test.csv"))

# === Plot Nemenyi heatmap ===
plt.figure(figsize=(12, 10))
sns.heatmap(nemenyi_result, annot=True, cmap="coolwarm", fmt=".2f", square=True,
            cbar_kws={'label': 'P-value (Nemenyi)'})
plt.title("Nemenyi Post-hoc Test (p-values)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "nemenyi_posthoc_heatmap.png"))
plt.close()