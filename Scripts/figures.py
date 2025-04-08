import os

import pandas as pd
import networkx as nx
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.pyplot import savefig


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

def get_dist(col, num=100, max=1.0):
    xs = np.linspace(0.0, max, num=num)
    ys = [(col >= threshold).sum() for threshold in xs]
    return xs, ys


def plot_centrality_dist(data, name, deriv=None):
    if name in ["Subgraph Centrality", "Out-degree Centrality", "In-degree Centrality", "Degree Centrality"]:
        dist_xs, dist_ys = get_dist(data, max=data.max())
    else:
        dist_xs, dist_ys = get_dist(data)

    fig_path = os.path.join("Figures", "CDF", f"{name}.pdf")
    if deriv is not None:
        if deriv == 1:
            deriv = first_derivative(dist_xs, dist_ys)
            fig_path = os.path.join("Figures", "CDF", "First derivative", f"{name}.pdf")
        elif deriv == 2:
            deriv = second_derivative(dist_xs, dist_ys)
            fig_path = os.path.join("Figures", "CDF", "Second derivative", f"{name}.pdf")
        # Normalize derivative values for colormap
        norm = Normalize(vmin=np.min(deriv), vmax=np.max(deriv))
        cmap = get_cmap('coolwarm_r')  # or 'viridis', 'plasma', et


    # Cumulative distribution
    plt.figure(figsize=(8,8))
    plt.plot(dist_xs, dist_ys)
    plt.xlabel("Threshold")
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.ylabel("Count(Centrality > Threshold)")
    plt.grid(linestyle=':')
    if deriv is not None:
        for i in range(len(dist_xs) - 1):
            plt.axvspan(dist_xs[i], dist_xs[i + 1],
                       color=cmap(norm(deriv[i])),
                       alpha=0.4,
                       linewidth=0)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(fig_path)


def comparison_figure():
    metrics = pd.read_csv(f"Metrics/metrics_centrality.csv")
    norm_deg_dist = get_dist(metrics["Norm. Degree Centrality"])
    deg_dist = get_dist(metrics["Degree Centrality"], max=metrics[f"Degree Centrality"].max())
    norm_in_deg_dist = get_dist(metrics["Norm. In-degree Centrality"])
    in_deg_dist = get_dist(metrics["In-degree Centrality"], max=metrics[f"In-degree Centrality"].max())
    norm_out_deg_dist = get_dist(metrics["Norm. Out-degree Centrality"])
    out_deg_dist = get_dist(metrics[f"Out-degree Centrality"], max=metrics[f"Out-degree Centrality"].max())

    ## Comparison figure
    plt.figure(figsize=(10, 8))
    plt.subplot(331)
    plt.plot(deg_dist[0], deg_dist[1], color='tab:orange')
    plt.xlabel("Threshold")
    plt.ylabel("Count(Centrality > Threshold)")
    plt.xticks([0, 2, 4, 6, 8, 10, 11], [0, 2, 4, 6, 8, 10, 11])
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
    plt.xticks([0, 2, 4, 6, 8, 10, 11], [0, 2, 4, 6, 8, 10, 11])
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
    plt.xticks([0, 2, 4, 6, 8, 10, 11], [0, 2, 4, 6, 8, 10, 11])
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
    plt.suptitle(f"Comparison of Absolute and Normalized\n Cumulative Distributions of Degree centralities\n")
    plt.tight_layout()
    plt.savefig("Figures/Comparison.pdf")


def scale_free_figure():
    scale_free = pd.read_csv("Metrics/scale_free_proportion.csv")
    deg_scale = scale_free["Degree Centrality"]
    in_deg_scale = scale_free["In-degree Centrality"]
    out_deg_scale = scale_free["Out-degree Centrality"]

    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.plot(deg_scale)
    plt.title("Degree")
    plt.xlim((0, 12))
    plt.ylim((0.0, 0.5))
    plt.ylabel("Proportion of nodes, P(k)")
    plt.xlabel("Degree, k")
    plt.tight_layout()

    plt.subplot(1,3,2)
    plt.plot(in_deg_scale)
    plt.xlabel("Degree, k")
    plt.xlim((0, 12))
    plt.ylim((0.0, 0.5))
    plt.title("In-degree")
    plt.tight_layout()

    plt.subplot(1,3,3)
    plt.plot(out_deg_scale)
    plt.xlabel("Degree, k")
    plt.xlim((0, 12))
    plt.ylim((0.0, 0.5))
    plt.title("Out-degree")
    plt.tight_layout()

    plt.suptitle("Proportion of nodes P(k) for a specific degree k")
    plt.savefig("Figures/ScaleFree.pdf")

def clustering_scatterplot(centrality, clustering, name, microservices):

    plt.figure(figsize=(8,8))
    plt.scatter(clustering, centrality)
    plt.title(f"Scatter plot of {name} vs. Clustering coeff.")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel(name)
    ax = plt.gca()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    ax.invert_xaxis()

    # Compute axis limits
    min_x, max_x = 0.0, 1.0
    min_y, max_y = 0.0, 1.0

    # Create diagonal lines with y - x = constant (adjusted for visual top-right direction)
    x_vals = np.linspace(min_x, max_x, 500)
    offsets = np.linspace((min_y - max_x), (max_y - min_x), 25)

    for offset in offsets:
        y_vals = x_vals + offset
        ax.plot(x_vals, y_vals, linestyle='--', color='gray', linewidth=0.5)
    #
    # for i, label in enumerate(microservices):
    #     ax.annotate(label,
    #                 (clustering[i], centrality[i]),
    #                 textcoords="offset points",  # offset the text slightly
    #                 xytext=(0, 5),
    #                 ha='center',
    #                 fontsize=8)

    plt.savefig(f"Figures/ClusteringScatter/ClusteringScatter_{name}.pdf")


def call_graphs():
    for f in os.scandir("Raw"):
        with open(f, 'r') as fi:
            g = json.load(fi)
        G = nx.node_link_graph(g, edges="edges", nodes="nodes", name="name", source="sender", target="receiver",
                               multigraph=False, directed=True)
        G.remove_nodes_from(["user"])
        G.remove_edges_from(nx.selfloop_edges(G))
        name = f.name.replace("_gwcc_noDB.json", "")
        plt.figure(figsize=(16, 12))
        nx.draw_networkx(G, pos=nx.fruchterman_reingold_layout(G))
        plt.tight_layout()
        plt.savefig(f"Figures/SDGs/{name}_sdg.pdf")
        plt.close()



if __name__ == "__main__":
    comparison_figure()
    scale_free_figure()
    metrics = pd.read_csv(f"Metrics/metrics_centrality.csv")
    # call_graphs()
    for col in metrics.columns:
        if col in ["MS_system", "Microservice"]:
            continue
        plot_centrality_dist(metrics[col], col)
        plot_centrality_dist(metrics[col], col, deriv=1)
        plot_centrality_dist(metrics[col], col, deriv=2)
        if col != "Clustering Coefficient":
            clustering_scatterplot(metrics[col], metrics["Clustering Coefficient"],
                                   col, metrics["Microservice"])
