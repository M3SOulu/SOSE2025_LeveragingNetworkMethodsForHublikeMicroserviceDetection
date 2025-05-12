import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

metrics = pd.read_csv("Metrics/CentralityMetrics.csv")
centrality_cols = [col for col in metrics.columns if col not in ['MS_system',
                                                                 "Microservice",
                                                                 'Clustering Coefficient',
                                                                 'Degree',
                                                                 'In-degree',
                                                                 'Out-degree']]

sorted_microservices = {}
metrics["Microservice"] = metrics["MS_system"] + "_" + metrics["Microservice"]
for centrality in centrality_cols:
    # Compute difference
    metrics[f'diff_{centrality}'] = metrics[centrality] - metrics['Clustering Coefficient']

    # Sort by the difference (descending or ascending depending on your preference)
    sorted_df = metrics.sort_values(by=f'diff_{centrality}', ascending=False)

    # Store only the "Microservice" column in the dictionary (or keep full row if needed)
    sorted_microservices[centrality] = sorted_df["Microservice"].tolist()

    # Scatter plot for RQ1.3
    plt.figure(figsize=(8,8))
    plt.scatter(metrics["Clustering Coefficient"], metrics[centrality])
    # plt.title(f"Scatter plot of {name} vs. Clustering coeff.")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel(centrality)
    ax = plt.gca()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Compute axis limits
    min_x, max_x = 0.0, 1.0
    min_y, max_y = 0.0, 1.0

    # Create diagonal lines with y - x = constant (adjusted for visual top-right direction)
    x_vals = np.linspace(min_x, max_x, 500)
    offsets = np.linspace((min_y - max_x), (max_y - min_x), 21)

    for offset in offsets:
        y_vals = x_vals + offset
        y_vals = np.clip(y_vals, min=0.0, max=1.0)
        ax.plot(x_vals, y_vals, linestyle='--', color='gray', linewidth=0.5)

    # Parameters for gradient
    N = 300  # Grid resolution
    steps = 20  # Number of color bands (sharper = fewer, distinct colors)

    # Generate banded gradient: diagonal ↘ direction
    gradient = np.fromfunction(lambda i, j: np.floor((i - j) * steps / N + steps / 2), (N, N))

    # Create sharp red-to-blue colormap with discrete colors
    colors = np.linspace(0, 1, steps)
    cmap = ListedColormap(plt.cm.bwr(colors))  # 'bwr' is red → white → blue, perfect for contrast

    # Overlay step-wise diagonal bands
    ax.imshow(gradient,
              extent=[0, 1, 0, 1],
              origin='lower',
              cmap=cmap,
              interpolation='nearest',  # Ensures hard edges
              alpha=0.3,
              aspect='auto',
              zorder=0)

    plt.savefig(f"Figures/ClusteringScatter/ClusteringScatter_{centrality}.pdf", bbox_inches='tight')

# Convert results to DataFrame for better display (optional)
results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in sorted_microservices.items()]))

results_df.to_csv("Metrics/ClusteringHubStrength.csv", index=False)