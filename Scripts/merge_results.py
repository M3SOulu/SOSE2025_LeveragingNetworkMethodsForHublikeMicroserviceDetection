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
    centrality_cols = [col for col in metrics_df.columns if col not in ['MS_system',
                                                                     "Microservice",
                                                                     'Clustering Coefficient',
                                                                        'Degree Centrality',
                                                                        'In-degree Centrality',
                                                                        'Out-degree Centrality']]
    int_df = metrics_df[["MS_system", "Microservice"]]
    for centrality in centrality_cols:
        # Compute difference
        int_df[f'Int. Clustering & {centrality}'] = metrics_df[centrality] - metrics_df['Clustering Coefficient']
        int_df[f'Int. Clustering & {centrality}'] = int_df[f'Int. Clustering & {centrality}'].clip(lower=-1.0, upper=1.0)
    # Compute quantiles from 1% to 100%
    low, medium, high = arcan_threshold("Degree Centrality", metrics_df)
    print(low, medium, high)
    low_t, medium_t, high_t = arcan_threshold("Norm. Degree Centrality", metrics_df)
    print(low_t, medium_t, high_t)

    ref_df = metrics_df[["MS_system", "Microservice", "Degree Centrality", "Norm. Degree Centrality"]]

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
    merged_df["Arcan_abs"] = merged_df["Degree Centrality"] >= high
    merged_df["Arcan_norm"] = merged_df["Norm. Degree Centrality"] >= high_t
    del merged_df["Degree Centrality"]
    del merged_df["Norm. Degree Centrality"]

    # Scale-free test failed, so no hubs for scale-free
    merged_df["ScaleFree"] = None
    merged_df = pd.merge(merged_df, int_df, on=["MS_system", "Microservice"], how="left")
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


def arcan_threshold(metric_col, metrics_df):

    values = metrics_df[metric_col].dropna().values

    # Step 1: Compute 100 quantile steps
    percentiles = np.linspace(0.01, 1, 100)
    qf = np.percentile(values, percentiles * 100)

    # Step 2: Frequency distribution
    freq = pd.Series(qf).value_counts().sort_index()
    freq_mid = freq.median()

    # Step 3: Determine v_mid (start of meaningful variability)
    sorted_qf = pd.Series(qf).sort_values().unique()

    v_mid = None
    for v in sorted_qf:
        if freq.get(v, 0) <= freq_mid:
            right_side = sorted_qf[sorted_qf >= v]
            if all(freq.get(w, 0) <= freq_mid for w in right_side):
                v_mid = v
                break

    # Fallback if no v_mid found
    if v_mid is None:
        cropped = values
    else:
        cropped = values[values >= v_mid]
    if len(cropped) < 3:
        # If cropped set too small, fall back to original values
        cropped = values

    # Step 5: Compute thresholds
    low = np.percentile(cropped, 25)
    medium = np.percentile(cropped, 50)
    high = np.percentile(cropped, 75)

    return low, medium, high


if __name__ == "__main__":
    main()