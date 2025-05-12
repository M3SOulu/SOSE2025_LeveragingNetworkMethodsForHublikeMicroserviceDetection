import json
from itertools import combinations

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    # Start by loading Kirkley
    with open("Results/RQ1/Kirkley.json", 'r') as f:
        results = json.load(f)

    # Insert results from the clustering scatterplots
    clustering = pd.read_csv("Results/RQ1/ClusteringHubsAgreement.csv", delimiter=";")
    for centrality in clustering.columns:
        for item in clustering[centrality]:
            if pd.isna(item):
                continue
            parts = item.split("_")
            system = "_".join(parts[:-1])
            service = parts[-1]
            l = results[system].setdefault(f"Clustering & {centrality}", [])
            l.append(service)


    metrics_df = pd.read_csv("Metrics/CentralityMetrics.csv")
    centrality_cols = [col for col in metrics_df.columns if col not in ['MS_system',
                                                                     "Microservice",
                                                                     'Clustering Coefficient',
                                                                        'Degree',
                                                                        'In-degree',
                                                                        'Out-degree']]
    int_df = metrics_df[["MS_system", "Microservice"]]
    for centrality in centrality_cols:
        # Compute difference
        int_df[f'Int. Clustering & {centrality}'] = metrics_df[centrality] - metrics_df['Clustering Coefficient']
        int_df[f'Int. Clustering & {centrality}'] = int_df[f'Int. Clustering & {centrality}'].clip(lower=-1.0, upper=1.0)
    # Compute quantiles from 1% to 100%
    low, medium, high = arcan_threshold("Degree", metrics_df)
    print(low, medium, high)
    low_t, medium_t, high_t = arcan_threshold("Degree Centrality", metrics_df)
    print(low_t, medium_t, high_t)

    ref_df = metrics_df[["MS_system", "Microservice", "Degree", "Degree Centrality"]]

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
    merged_df["Arcan_abs"] = merged_df["Degree"] >= high
    merged_df["Arcan_norm"] = merged_df["Degree Centrality"] >= high_t
    del merged_df["Degree"]
    del merged_df["Degree Centrality"]

    # Scale-free test failed, so no hubs for scale-free
    merged_df["ScaleFree"] = None
    merged_df = pd.merge(merged_df, int_df, on=["MS_system", "Microservice"], how="left")
    manual = pd.read_csv("Results/RQ3/ManualValidation.csv")
    merged_df = pd.merge(merged_df, manual, how = "left")
    merged_df.to_csv("Results/RQ1/HubsAll.csv", index=False, header=True)

    # Select only boolean columns
    bool_cols = merged_df.select_dtypes(include=bool).columns

    filtered_df = merged_df[merged_df[bool_cols].any(axis=1)]
    filtered_df.to_csv("Results/RQ1/HubsTrue.csv")
    count_df = merged_df[bool_cols]
    count_df = count_df.sum()
    count_df.to_csv("Results/RQ1/HubCounts.csv")

    # Dictionary to store results

    in_cols = ["AVG_in_degree", "LOUBAR_in_degree", "CM_in_degree",
               "ER_in_degree", "Clustering & In-degree Centrality", "Clustering & Eigenvector Centrality","Clustering & Authority Score"]
    out_cols = ["AVG_out_degree", "LOUBAR_out_degree", "CM_out_degree",
               "ER_out_degree", "Clustering & Out-degree Centrality", "Clustering & Hub Score"]
    total_cols = ["Clustering & Degree Centrality", "Arcan_abs", "Arcan_norm",
                  "Clustering & Betweenness Centrality", "Clustering & Closeness Centrality",
                   "Clustering & PageRank Centrality"]

    agreement(in_cols, merged_df, "incoming")
    agreement(out_cols, merged_df, "outgoing")
    agreement(total_cols, merged_df, "all")
    agreement(bool_cols, merged_df, "everything")

    precision_results = [precision(merged_df, method) for method in [*in_cols, *out_cols, *total_cols]]
    precision_results.insert(0, ("Method", "Precision (Infra is TP)", "Precision (No Infra)", "Precision (Infra is FP)"))
    precision_df = pd.DataFrame(precision_results)
    precision_df.to_csv("Results/RQ3/Precision.csv", index=False, header=False)


def precision(data_df, method):
    # Take only the current method data
    comp_df = data_df[["MS_system", "Microservice", method, "Manual Validation"]]

    # Keep only the TP+FP services (all detected Hubs)
    comp_df = comp_df[comp_df[method]]

    # Consider Infrastructural Hubs as True Positives
    comp_df["Hublike Full"] = comp_df["Manual Validation"].map({"True": True, "False": False,
                                                          "Infra": True})
    # Compute precision with infrastructural hubs as True
    TP = int(comp_df["Hublike Full"].sum())
    P = int(comp_df[method].sum())
    precision_true = TP / P if P != 0 else None
    print(f"Precision: {method}, Infra is True, {TP=}, {P=}, precision={precision_true}")

    # Consider Infrastructural Hubs as True Negatives
    comp_df["Hublike Full"] = comp_df["Manual Validation"].map({"True": True, "False": False,
                                                          "Infra": False})
    # Compute precision with infrastructural hubs as False
    TP = int(comp_df["Hublike Full"].sum())
    P = int(comp_df[method].sum())
    precision_false = TP / P if P != 0 else None
    print(f"Precision: {method}, Infra is False, {TP=}, {P=}, precision={precision_false}")

    # Filter infrastructural hubs
    comp_df = comp_df[comp_df["Manual Validation"] != "Infra"]
    # Compute precision without infrastructural hubs
    TP = int(comp_df["Hublike Full"].sum())
    P = int(comp_df[method].sum())
    precision_non_infra = TP / P if P != 0 else None
    print(f"Precision: {method}, Infra is ignored, {TP=}, {P=}, precision={precision_non_infra}")
    return method, precision_true, precision_non_infra, precision_false

def agreement(in_cols, merged_df, name):
    agreements = {}
    # Compute pairwise agreement
    for col1, col2 in combinations(in_cols, 2):
        intersection = ((merged_df[col1]) & (merged_df[col2])).sum()
        union = ((merged_df[col1]) | (merged_df[col2])).sum()
        agreements[(col1, col2)] = intersection / union if union > 0 else 0
    # Convert to DataFrame for display
    agreement_df = pd.DataFrame([
        {"Column 1": k[0], "Column 2": k[1], "Jaccard Index": v}
        for k, v in agreements.items()
    ])

    # Step 2: Convert to counts of True/False per item
    # Each row will have: [count_false, count_true]
    rating_counts = merged_df[in_cols].apply(lambda row: pd.Series([
        (~row).sum(),  # count of False
        row.sum()      # count of True
    ]), axis=1)

    rating_counts.columns = ['False', 'True']
    kappa = fleiss_kappa(rating_counts)

    agreement_df.to_csv(f"Results/RQ2/HubJaccard_{name}.csv", index=False, header=True)
    # Step 1: Pivot agreement_df into square matrix
    heatmap_data = agreement_df.pivot(index="Column 1", columns="Column 2", values="Jaccard Index")
    # Step 2: Make the matrix symmetric by filling in the lower triangle
    # Optionally include diagonal = 1.0
    all_cols = sorted(set(heatmap_data.columns).union(set(heatmap_data.index)))
    heatmap_data = heatmap_data.reindex(index=all_cols, columns=all_cols)
    rename_method = {"AVG_in_degree": "Avg_in", "LOUBAR_in_degree": "Loubar_in",
                     "CM_in_degree": "CM_in", "ER_in_degree": "EM_in",
                     "Clustering & In-degree Centrality": "Cl. & In-degree c.",
                     "Clustering & Eigenvector Centrality": "Cl. & Eigenvector c.",
                     "Clustering & Authority Score": "Cl. & Authority sc.",
                     "AVG_out_degree": "Avg_out", "LOUBAR_out_degree": "Loubar_out",
                     "CM_out_degree": "CM_out", "ER_out_degree": "EM_out",
                     "Clustering & Out-degree Centrality": "Cl. & Out-degree c.",
                     "Clustering & Hub Score": "Cl. & Hub sc.",
                     "Clustering & Degree Centrality": "Cl. & Degree c.",
                     "Clustering & Betweenness Centrality": "Cl. & Betweenness c.",
                     "Clustering & Closeness Centrality": "Cl. & Closeness c.",
                     "Clustering & PageRank Centrality": "Cl. & PageRank c."}
    for col1 in all_cols:
        for col2 in all_cols:
            if pd.isna(heatmap_data.loc[col1, col2]) and not pd.isna(heatmap_data.loc[col2, col1]):
                heatmap_data.loc[col1, col2] = heatmap_data.loc[col2, col1]
            elif col1 == col2:
                heatmap_data.loc[col1, col2] = 1.0  # full agreement with self
    # Step 3: Plot the heatmap
    plt.figure(figsize=(10, 10))
    heatmap_data = heatmap_data.rename(index=rename_method, columns=rename_method)
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": 14},  # Font size for cell annotations
        cbar_kws={'label': 'Agreement', 'shrink': 0.6}  # Shorten colorbar
    )
    # Make method names (tick labels) larger
    plt.xticks(fontsize=16, rotation=45, ha='right')  # X-axis: method names
    plt.yticks(fontsize=16, rotation=0)  # Y-axis: method names
    plt.xlabel("")
    plt.ylabel("")
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=14)  # Tick label font size
    colorbar.ax.set_ylabel("Jaccard Index", fontsize=16)  # Label font size

    # if name == "everything":
    #     plt.title(f"Jaccard Index Between all Hub detectors\nFleiss Kappa = {kappa:.4f}")
    # else:
    #     plt.title(f"Jaccard Index Between Hub detectors based on {name} connections\nFleiss Kappa = {kappa:.4f}")
    plt.tight_layout()
    plt.savefig(f"Figures/HubJaccard_{name}.pdf", bbox_inches='tight')

    # Step 2: Convert to counts of True/False per item
    # Each row will have: [count_false, count_true]
    rating_counts = merged_df[in_cols].apply(lambda row: pd.Series([
        (~row).sum(),  # count of False
        row.sum()      # count of True
    ]), axis=1)

    rating_counts.columns = ['False', 'True']
    # Step 4: Compute kappa
    kappa = fleiss_kappa(rating_counts)
    print(f"Fleiss' Kappa: {kappa:.4f}")

def fleiss_kappa(table):
    """
    Compute Fleiss' kappa for a category count table (N_items x N_categories)
    """
    N, k = table.shape
    n = np.sum(table.values[0])  # assumes constant number of raters

    # Proportion of all assignments to each category
    p = np.sum(table, axis=0) / (N * n)

    # Agreement per item
    P = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))

    # Mean of P, and expected agreement
    P_bar = P.mean()
    P_e = np.sum(p ** 2)

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa

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