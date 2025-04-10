import json
import pandas as pd

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

    ref_df = pd.read_csv("Metrics/metrics_centrality.csv")
    ref_df = ref_df[["MS_system", "Microservice"]]
    # Step 1: Flatten JSON into long format
    rows = []
    all_methods = set()
    for system, methods in kirkley.items():
        for method, services in methods.items():
            all_methods.add(method)
            for service in services:
                rows.append((system, service, method))

    df = pd.DataFrame(rows, columns=["MS_system", "Microservice", "Method"])



    # Step 2: Pivot to wide format with methods as boolean flags
    method_df = pd.crosstab(index=[df["MS_system"], df["Microservice"]],
                            columns=df["Method"]).astype(bool).reset_index()

    # Step 3: Ensure all method columns exist
    for method in all_methods:
        if method not in method_df.columns:
            method_df[method] = False

    # Step 3: Merge with reference DataFrame to ensure all services are included
    merged_df = pd.merge(ref_df, method_df, on=["MS_system", "Microservice"], how="left")

    # Step 4: Fill NaN (from missing entries) with False
    method_cols = [col for col in merged_df.columns if col not in ["MS_system", "Microservice"]]
    merged_df[method_cols] = merged_df[method_cols].fillna(False)
    merged_df.to_csv("Results/HubTable.csv", index=False, header=True)

if __name__ == "__main__":
    main()