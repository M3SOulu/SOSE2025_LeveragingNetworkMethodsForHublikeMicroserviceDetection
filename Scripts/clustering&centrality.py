import pandas as pd

metrics = pd.read_csv("Metrics/metrics_centrality.csv")
centrality_cols = [col for col in metrics.columns if col not in ['MS_system',
                                                                 "Microservice",
                                                                 'Clustering Coefficient']]

sorted_microservices = {}

for centrality in centrality_cols:
    # Compute difference
    metrics[f'diff_{centrality}'] = metrics[centrality] - metrics['Clustering Coefficient']

    # Sort by the difference (descending or ascending depending on your preference)
    sorted_df = metrics.sort_values(by=f'diff_{centrality}', ascending=False)

    # Store only the "Microservice" column in the dictionary (or keep full row if needed)
    sorted_microservices[centrality] = sorted_df["Microservice"].tolist()

# Convert results to DataFrame for better display (optional)
results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in sorted_microservices.items()]))

results_df.to_csv("ClusteringRank.csv", index=False)