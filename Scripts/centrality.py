import os
import json

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def compute_centrality():
    all_dfs = []
    for f in os.scandir("Raw/graph"):
        with open(f, 'r') as fi:
            g = json.load(fi)
        G = nx.node_link_graph(g, edges="edges", nodes="nodes", name="name", source="sender", target="receiver",
                               multigraph=False, directed=True)
        G.remove_nodes_from(["user"])
        G.remove_edges_from(nx.selfloop_edges(G))
        name = f.name.replace("_gwcc_noDB.json", "")
        plt.figure(figsize=(16, 12))
        nx.draw_networkx(G, pos=nx.arf_layout(G))
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f"Figures/SDGs/{name}_sdg.pdf")
        plt.close()
        graph_df = pd.DataFrame(columns=["MS_system", "Microservice"])
        for node in G.nodes:
            graph_df.loc[len(graph_df)] = [name, node]
        norm_degree_centrality = nx.degree_centrality(G)
        degree_centrality = {k: int(v * (len(G.nodes) - 1)) for k, v in norm_degree_centrality.items()}
        norm_in_degree_centrality = nx.in_degree_centrality(G)
        in_degree_centrality = {k: int(v * (len(G.nodes) - 1)) for k, v in norm_in_degree_centrality.items()}
        norm_out_degree_centrality = nx.out_degree_centrality(G)
        out_degree_centrality = {k: int(v * (len(G.nodes) - 1)) for k, v in norm_out_degree_centrality.items()}
        clustering_coefficient = nx.clustering(G)

        graph_df[f"Degree"] = graph_df["Microservice"].map(degree_centrality)
        graph_df[f"Degree Centrality"] = graph_df["Microservice"].map(norm_degree_centrality)
        graph_df[f"In-degree"] = graph_df["Microservice"].map(in_degree_centrality)
        graph_df[f"In-degree Centrality"] = graph_df["Microservice"].map(norm_in_degree_centrality)
        graph_df[f"Out-degree"] = graph_df["Microservice"].map(out_degree_centrality)
        graph_df[f"Out-degree Centrality"] = graph_df["Microservice"].map(norm_out_degree_centrality)
        graph_df[f"Eigenvector Centrality"] = graph_df["Microservice"].map(
            nx.eigenvector_centrality(G, max_iter=1000))
        graph_df[f"Betweenness Centrality"] = graph_df["Microservice"].map(nx.betweenness_centrality(G))
        graph_df[f"Closeness Centrality"] = graph_df["Microservice"].map(nx.closeness_centrality(G))
        graph_df[f"PageRank Centrality"] = graph_df["Microservice"].map(nx.pagerank(G))
        hubs, authorities = nx.hits(G)
        graph_df[f"Hub Score"] = graph_df["Microservice"].map(hubs)
        graph_df[f"Authority Score"] = graph_df["Microservice"].map(authorities)
        graph_df["Clustering Coefficient"] = graph_df["Microservice"].map(clustering_coefficient)

        all_dfs.append(graph_df)

    all_dfs = sorted(all_dfs, key=lambda d: d["MS_system"].iloc[0].casefold())
    df = pd.concat(all_dfs)
    df.to_csv(os.path.join("Metrics", "CentralityMetrics.csv"), index=False, header=True)

    ## Comparison figure
    def get_dist(col, num=100, max=1.0):
        xs = np.linspace(0.0, max, num=num)
        ys = [(col >= threshold).sum() for threshold in xs]
        return xs, ys
    norm_deg_dist = get_dist(df["Degree Centrality"])
    deg_dist = get_dist(df["Degree"], max=df[f"Degree"].max())
    plt.figure(figsize=(10, 2.5))
    plt.subplot(131)
    plt.plot(deg_dist[0], deg_dist[1], color='tab:orange')
    plt.xlabel("Threshold")
    plt.ylabel("Count(Centrality > Threshold)")
    plt.xticks([0, 2, 4, 6, 8, 10, 11], [0, 2, 4, 6, 8, 10, 11])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Degree")
    plt.subplot(132)
    plt.plot(norm_deg_dist[0], norm_deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Degree Centrality")
    plt.subplot(133)
    plt.plot(norm_deg_dist[0], norm_deg_dist[1])
    plt.plot(norm_deg_dist[0], deg_dist[1])
    plt.xlabel("Threshold")
    plt.xticks([0.0, 1.0], ["min", "max"])
    plt.yticks([0, 50, 100, 150, 200, 250], [0, 50, 100, 150, 200, 250])
    plt.title("Overlaid")
    plt.tight_layout()
    plt.savefig("Figures/Comparison.pdf", bbox_inches='tight')


if __name__ == "__main__":
    compute_centrality()
