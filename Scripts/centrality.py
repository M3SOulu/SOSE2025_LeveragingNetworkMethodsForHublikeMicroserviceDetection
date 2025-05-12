import os
import json

import networkx as nx
import pandas as pd


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
    df_db = pd.concat(all_dfs)
    df_db.to_csv(os.path.join("Metrics", "CentralityMetrics.csv"), index=False, header=True)


if __name__ == "__main__":
    compute_centrality()
