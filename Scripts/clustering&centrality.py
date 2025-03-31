import pandas as pd

metrics = pd.read_csv("Metrics/metrics_centrality.csv")
centrality_cols = [col for col in metrics.columns if col not in ['MS_system',
                                                                 "Microservice",
                                                                 'Clustering Coefficient']]


results= {}
for centrality in centrality_cols:
    # Compute thresholds
    centrality_thresh = metrics[centrality].quantile(0.75)
    clustering_thresh = metrics['Clustering Coefficient'].quantile(0.25)

    # Apply filter
    filtered = metrics[
        (metrics[centrality] >= centrality_thresh) &
        (metrics['Clustering Coefficient'] <= clustering_thresh)
        ]

    # Store the matching nodes
    results[centrality] = filtered['Microservice'].tolist()

# Convert results to DataFrame for better display (optional)
results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))

results_df.to_csv("ClusteringPercentiles.csv", index=False)