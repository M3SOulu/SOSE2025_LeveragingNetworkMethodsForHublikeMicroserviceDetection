import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_dist(df, col, max, num=100):
    xs = np.linspace(0.0, max, num=num)
    ys = [(df[col] >= threshold).sum() for threshold in xs]
    return xs, ys


def plot_centrality_dist(metrics, DB: bool):
    metrics = metrics[metrics["DB"] == DB]
    norm_deg_dist = get_dist(metrics, "Norm. Degree Centrality", 1.0)
    deg_dist = get_dist(metrics, "Degree Centrality", metrics["Degree Centrality"].max())
    norm_in_deg_dist = get_dist(metrics, "Norm. In-degree Centrality", 1.0)
    in_deg_dist = get_dist(metrics, "In-degree Centrality", metrics["In-degree Centrality"].max())
    norm_out_deg_dist = get_dist(metrics, "Norm. Out-degree Centrality", 1.0)
    out_deg_dist = get_dist(metrics, "Out-degree Centrality", metrics["Out-degree Centrality"].max())
    bet_dist = get_dist(metrics, "Betweenness Centrality", 1.0)
    eig_dist = get_dist(metrics, "Eigenvector Centrality", 1.0)

    plt.figure(figsize=(8,7))
    plt.subplot(321)
    plt.plot(deg_dist[0], deg_dist[1])
    plt.title("Degree Centrality")
    plt.subplot(322)
    plt.plot(norm_deg_dist[0], norm_deg_dist[1])
    plt.title("Norm. Degree Centrality")
    plt.subplot(323)
    plt.plot(in_deg_dist[0], in_deg_dist[1])
    plt.title("In-degree Centrality")
    plt.subplot(324)
    plt.plot(norm_in_deg_dist[0], norm_in_deg_dist[1])
    plt.title("Norm. In-degree Centrality")
    plt.subplot(325)
    plt.plot(out_deg_dist[0], out_deg_dist[1])
    plt.title("Out-degree Centrality")
    plt.subplot(326)
    plt.plot(norm_out_deg_dist[0], norm_out_deg_dist[1])
    plt.title("Norm. Out-degree Centrality")
    plt.suptitle(f"Comparison of Absolute and Normalized Degree centralities (DB={DB})")
    plt.tight_layout()
    plt.savefig(f"ComparsionDB={DB}.png")

    plt.figure(figsize=(10,5))
    plt.subplot(231)
    plt.plot(norm_deg_dist[0], norm_deg_dist[1])
    plt.title("Norm. Degree Centrality")
    plt.subplot(232)
    plt.plot(norm_in_deg_dist[0], norm_in_deg_dist[1])
    plt.title("Norm. In-degree Centrality")
    plt.subplot(233)
    plt.plot(norm_out_deg_dist[0], norm_out_deg_dist[1])
    plt.title("Norm. Out-degree Centrality")
    plt.subplot(234)
    plt.plot(bet_dist[0], bet_dist[1])
    plt.title("Betweenness Centrality")
    plt.subplot(235)
    plt.plot(eig_dist[0], eig_dist[1])
    plt.title("Eigenvector Centrality")
    plt.suptitle(f"Cumulative distributions of centrality metrics (DB={DB})")
    plt.tight_layout()
    plt.savefig(f"allMetricsDB={DB}.png")


metrics = pd.read_csv("metrics_centrality.csv")
plot_centrality_dist(metrics, DB=True)
plot_centrality_dist(metrics, DB=False)
