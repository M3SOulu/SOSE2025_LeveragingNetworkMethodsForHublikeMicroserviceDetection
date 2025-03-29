import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

def first_derivative(x, y):
    x = np.array(x)
    y = np.array(y)
    dy = np.zeros_like(y)
    dx = np.diff(x)

    dy[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])  # central
    dy[0] = (y[1] - y[0]) / dx[0]                  # forward
    dy[-1] = (y[-1] - y[-2]) / dx[-1]              # backward
    return dy

def second_derivative(xs, ys):
    h = xs[1] - xs[0]
    n = len(ys)
    second_deriv = [0.0] * n  # Initialize result list

    # Forward difference for the first point
    second_deriv[0] = (ys[2] - 2 * ys[1] + ys[0]) / h ** 2

    # Central difference for interior points
    for i in range(1, n - 1):
        second_deriv[i] = (ys[i + 1] - 2 * ys[i] + ys[i - 1]) / h ** 2

    # Backward difference for the last point
    second_deriv[-1] = (ys[-1] - 2 * ys[-2] + ys[-3]) / h ** 2

    return np.array(second_deriv)

def get_dist(df, col, max, num=100):
    xs = np.linspace(0.0, max, num=num)
    ys = [(df[col] >= threshold).sum() for threshold in xs]
    return xs, ys


def plot_centrality_dist(DB: bool):
    # metrics = metrics[metrics[f"Betweenness Centrality (DB={DB})"] > 0.0001]
    # metrics = metrics[metrics[f"Eigenvector Centrality (DB={DB})"] > 0.0001]
    metrics = pd.read_csv(f"metrics_centrality_db_{DB}.csv")
    norm_deg_dist = get_dist(metrics, f"Norm. Degree Centrality", 1.0)
    deg_dist = get_dist(metrics, f"Degree Centrality", metrics[f"Degree Centrality"].max())
    norm_in_deg_dist = get_dist(metrics, f"Norm. In-degree Centrality", 1.0)
    in_deg_dist = get_dist(metrics, f"In-degree Centrality", metrics[f"In-degree Centrality"].max())
    norm_out_deg_dist = get_dist(metrics, f"Norm. Out-degree Centrality", 1.0)
    out_deg_dist = get_dist(metrics, f"Out-degree Centrality", metrics[f"Out-degree Centrality"].max())
    bet_dist = get_dist(metrics, f"Betweenness Centrality", 1.0)
    closeness_dist = get_dist(metrics, f"Closeness Centrality", 1.0)
    eig_dist = get_dist(metrics, f"Eigenvector Centrality", 1.0)
    pagerank_dist = get_dist(metrics, f"PageRank Centrality", 1.0)
    hubs_dist = get_dist(metrics, f"Hub Score", 1.0)
    auth_dist = get_dist(metrics, f"Authority Score", 1.0)
    sub_dist = get_dist(metrics, f"Subgraph Centrality", metrics[f"Subgraph Centrality"].max())
    cluster_dist = get_dist(metrics, "Clustering", 1.0)

    norm_deg_deriv = second_derivative(*norm_deg_dist)

    # 3. Normalize derivative values for colormap
    norm = Normalize(vmin=np.min(norm_deg_deriv), vmax=np.max(norm_deg_deriv))
    cmap = get_cmap('cool_r')  # or 'viridis', 'plasma', et


    ## Comparison figure
    plt.figure(figsize=(14,12))
    plt.subplot(331)
    plt.plot(deg_dist[0], deg_dist[1], color='tab:orange')

    plt.xlabel("Threshold")
    plt.ylabel("Count(Centrality > Threshold)")
    plt.xticks([0,2,4,6,8,10,11],[0,2,4,6,8,10,11])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Degree Centrality")
    plt.subplot(332)
    plt.plot(norm_deg_dist[0], norm_deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Norm. Degree Centrality")
    plt.subplot(333)
    plt.plot(norm_deg_dist[0], norm_deg_dist[1])
    plt.plot(norm_deg_dist[0], deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 1.0], ["min", "max"])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Overlaid Degree Centrality")
    plt.subplot(334)
    plt.plot(in_deg_dist[0], in_deg_dist[1], color='tab:orange')
    plt.xlabel("Threshold")
    plt.ylabel("Count(Centrality > Threshold)")
    plt.xticks([0,2,4,6,8,10,11],[0,2,4,6,8,10,11])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("In-degree Centrality")
    plt.subplot(335)
    plt.plot(norm_in_deg_dist[0], norm_in_deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Norm. In-degree Centrality")
    plt.subplot(336)
    plt.plot(norm_in_deg_dist[0], norm_in_deg_dist[1])
    plt.plot(norm_in_deg_dist[0], in_deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 1.0], ["min", "max"])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Overlaid In-degree Centrality")
    plt.subplot(337)
    plt.plot(out_deg_dist[0], out_deg_dist[1], color='tab:orange')
    plt.xlabel("Threshold")
    plt.ylabel("Count(Centrality > Threshold)")
    plt.xticks([0,2,4,6,8,10,11],[0,2,4,6,8,10,11])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Out-degree Centrality")
    plt.subplot(338)
    plt.plot(norm_out_deg_dist[0], norm_out_deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Norm. Out-degree Centrality")
    plt.subplot(339)
    plt.plot(norm_out_deg_dist[0], norm_out_deg_dist[1])
    plt.plot(norm_out_deg_dist[0], out_deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 1.0], ["min", "max"])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Overlaid Out-degree Centrality")
    plt.suptitle(f"Comparison of Absolute and Normalized\n Cumulative Distributions of Degree centralities (DB={DB})\n")
    plt.tight_layout()
    plt.savefig(f"ComparisonDB={DB}.png")

    # Cumulative distribution
    plt.figure(figsize=(10,10))
    plt.plot(norm_deg_dist[0], norm_deg_dist[1])
    for i in range(len(norm_deg_dist[0]) - 1):
        plt.axvspan(norm_deg_dist[0][i], norm_deg_dist[0][i + 1],
                   color=cmap(norm(norm_deg_deriv[i])),
                   alpha=0.4,
                   linewidth=0)
    plt.xlabel("Threshold")
    plt.ylabel("Count(Centrality > Threshold)")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.title("Norm. Degree Centrality")
    plt.savefig(f"DegreeDB={DB}.png")

    plt.figure(figsize=(10,10))
    plt.plot(norm_in_deg_dist[0], norm_in_deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.title("Norm. In-degree Centrality")
    plt.savefig(f"InDegreeDB={DB}.png")

    plt.figure(figsize=(10,10))
    plt.plot(norm_out_deg_dist[0], norm_out_deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.title("Norm. Out-degree Centrality")
    plt.savefig(f"OutDegreeDB={DB}.png")

    plt.figure(figsize=(10,10))
    plt.plot(bet_dist[0], bet_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.ylabel("Count(Centrality > Threshold)")
    plt.title("Betweenness Centrality")
    plt.savefig(f"BetweennessDB={DB}.png")

    plt.figure(figsize=(10,10))
    plt.plot(eig_dist[0], eig_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.title("Eigenvector Centrality")
    plt.suptitle(f"Cumulative distributions of centrality metrics (DB={DB})")
    plt.tight_layout()
    plt.savefig(f"EigenvectorDB={DB}.png")

    plt.figure(figsize=(10,10))
    plt.plot(pagerank_dist[0], pagerank_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.title("PageRank Centrality")
    plt.suptitle(f"Cumulative distributions of centrality metrics (DB={DB})")
    plt.tight_layout()
    plt.savefig(f"PageRankDB={DB}.png")

    plt.figure(figsize=(10,10))
    plt.plot(closeness_dist[0], closeness_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.title("Closeness Centrality")
    plt.suptitle(f"Cumulative distributions of centrality metrics (DB={DB})")
    plt.tight_layout()
    plt.savefig(f"ClosenessDB={DB}.png")

    plt.figure(figsize=(10,10))
    plt.plot(hubs_dist[0], hubs_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.title("Hub Score")
    plt.tight_layout()
    plt.savefig(f"HubDB={DB}.png")

    plt.figure(figsize=(10,10))
    plt.plot(auth_dist[0], auth_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.title("Authority Score")
    plt.tight_layout()
    plt.savefig(f"AuthDB={DB}.png")
    plt.figure(figsize=(10,10))
    plt.plot(sub_dist[0], sub_dist[1])
    plt.xlabel("Threshold")
    plt.grid(linestyle=':')
    plt.title("Subgraph Centrality")
    plt.tight_layout()
    plt.savefig(f"SubgraphDB={DB}.png")

    plt.figure(figsize=(10,10))
    plt.plot(cluster_dist[0], cluster_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.grid(linestyle=':')
    plt.title("Clustering coefficient")
    plt.tight_layout()
    plt.savefig(f"Clustering={DB}.png")

    scale_free = pd.read_csv(f"scale_free_test_{DB}.csv")
    deg_scale = scale_free["Degree Centrality"]
    in_deg_scale = scale_free["In-degree Centrality"]
    out_deg_scale = scale_free["Out-degree Centrality"]

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(deg_scale)
    plt.title("Degree")
    plt.xlim((0, 12))
    plt.ylim((0.0, 0.5))
    plt.ylabel("Proportion of nodes, P(k)")
    plt.xlabel("Degree, k")

    plt.subplot(1,3,2)
    plt.plot(in_deg_scale)
    plt.xlabel("Degree, k")
    plt.xlim((0, 12))
    plt.ylim((0.0, 0.5))
    plt.title("In-degree")

    plt.subplot(1,3,3)
    plt.plot(out_deg_scale)
    plt.xlabel("Degree, k")
    plt.xlim((0, 12))
    plt.ylim((0.0, 0.5))
    plt.title("Out-degree")

    plt.suptitle("Proportion of nodes P(k) for a specific degree k")
    plt.savefig(f"scale_free_{DB}.png")


plot_centrality_dist(DB=True)
plot_centrality_dist(DB=False)
