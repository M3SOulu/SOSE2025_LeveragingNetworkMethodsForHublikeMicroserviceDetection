import json
from itertools import combinations

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    # Start by loading Kirkley
    with open("Results/Kirkley.json", 'r') as f:
        results = json.load(f)

    # Insert results from the clustering scatterplots
    clustering = pd.read_csv("Results/ClusteringRankAgreement.csv", delimiter=";")
    for centrality in clustering.columns:
        for item in clustering[centrality]:
            if pd.isna(item):
                continue
            parts = item.split("_")
            system = "_".join(parts[:-1])
            service = parts[-1]
            l = results[system].setdefault(f"Clustering & {centrality}", [])
            l.append(service)


    metrics_df = pd.read_csv("Metrics/metrics_centrality.csv")
    # Compute quantiles from 1% to 100%
    percentiles = np.linspace(0.01, 1, 100)
    metric_col = "Degree Centrality"
    qf = np.percentile(metrics_df[metric_col], percentiles * 100)

    # Frequency distribution
    freq = pd.Series(qf).value_counts().sort_index()
    freq_mid = freq.median()

    # Determine cropping threshold
    sorted_qf = pd.Series(qf).sort_values()
    v_mid = next(v for v in sorted_qf if freq.get(v, 0) <= freq_mid and
                 all(freq.get(w, 0) <= freq_mid for w in sorted_qf[sorted_qf >= v]))

    # Crop distribution
    cropped = metrics_df[metrics_df[metric_col] >= v_mid][metric_col]

    # Compute thresholds
    low = np.percentile(cropped, 25)
    medium = np.percentile(cropped, 50)
    high = np.percentile(cropped, 75)

    ref_df = metrics_df[["MS_system", "Microservice", "Degree Centrality"]]

    # Flatten JSON into long format
    rows = []
    all_methods = set()
    for system, methods in results.items():
        for method, services in methods.items():
            all_methods.add(method)
            for service in services:
                rows.append((system, service, method))

    df = pd.DataFrame(rows, columns=["MS_system", "Microservice", "Method"])

    # Pivot to wide format with methods as boolean flags
    method_df = pd.crosstab(index=[df["MS_system"], df["Microservice"]],
                            columns=df["Method"]).astype(bool).reset_index()

    # Ensure all method columns exist
    for method in all_methods:
        if method not in method_df.columns:
            method_df[method] = False

    # Merge with reference DataFrame to ensure all services are included
    merged_df = pd.merge(ref_df, method_df, on=["MS_system", "Microservice"], how="left")

    # Step 4: Fill NaN (from missing entries) with False
    method_cols = [col for col in merged_df.columns if col not in ["MS_system", "Microservice"]]
    merged_df[method_cols] = merged_df[method_cols].fillna(False)
    merged_df["Arcan"] = merged_df["Degree Centrality"] >= low
    del merged_df["Degree Centrality"]

    # Scale-free test failed, so no hubs for scale-free
    merged_df["ScaleFree"] = None
    merged_df.to_csv("Results/HubTable.csv", index=False, header=True)

    # Select only boolean columns
    bool_cols = merged_df.select_dtypes(include=bool).columns

    # Dictionary to store results
    agreements = {}

    # Compute pairwise agreement
    for col1, col2 in combinations(bool_cols, 2):
        agree = (merged_df[col1] == merged_df[col2]).sum()
        total = len(merged_df)
        agreements[(col1, col2)] = agree / total  # agreement ratio

    # Convert to DataFrame for display
    agreement_df = pd.DataFrame([
        {"Column 1": k[0], "Column 2": k[1], "Agreement": v}
        for k, v in agreements.items()
    ])

    print(agreement_df)
    agreement_df.to_csv("Results/Agreement.csv", index=False, header=True)
    # Step 1: Pivot agreement_df into square matrix
    heatmap_data = agreement_df.pivot(index="Column 1", columns="Column 2", values="Agreement")

    # Step 2: Make the matrix symmetric by filling in the lower triangle
    # Optionally include diagonal = 1.0
    all_cols = sorted(set(heatmap_data.columns).union(set(heatmap_data.index)))
    heatmap_data = heatmap_data.reindex(index=all_cols, columns=all_cols)
    for col1 in all_cols:
        for col2 in all_cols:
            if pd.isna(heatmap_data.loc[col1, col2]) and not pd.isna(heatmap_data.loc[col2, col1]):
                heatmap_data.loc[col1, col2] = heatmap_data.loc[col2, col1]
            elif col1 == col2:
                heatmap_data.loc[col1, col2] = 1.0  # full agreement with self

    # Step 3: Plot the heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(heatmap_data, annot=True, cmap="Blues", square=True, cbar_kws={'label': 'Agreement'})
    plt.title("Pairwise Agreement Between Hub detectors")
    plt.tight_layout()
    plt.savefig("Figures/HubAgreement.pdf")

if __name__ == "__main__":
    main()