import json
import pandas as pd
import numpy as np

def main():
    # Start by loading Kirkley
    with open("Results/Kirkley.json", 'r') as f:
        kirkley = json.load(f)
    # Scale-free test failed, so no hubs for scale-free
    for system in kirkley:
        kirkley[system]["scale-free"] = []
    # Insert results from the clustering scatterplots
    clustering = pd.read_csv("Results/ClusteringRankAgreement.csv", delimiter=";")
    for centrality in clustering.columns:
        for item in clustering[centrality]:
            if pd.isna(item):
                continue
            parts = item.split("_")
            system = "_".join(parts[:-1])
            service = parts[-1]
            l = kirkley[system].setdefault(f"Clustering & {centrality}", [])
            l.append(service)
    # Save merged results
    with open("Results/Hubs.json", 'w') as f:
        json.dump(kirkley, f, indent=4)


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
    for system, methods in kirkley.items():
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
    merged_df.to_csv("Results/HubTable.csv", index=False, header=True)

if __name__ == "__main__":
    main()