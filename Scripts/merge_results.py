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

if __name__ == "__main__":
    main()