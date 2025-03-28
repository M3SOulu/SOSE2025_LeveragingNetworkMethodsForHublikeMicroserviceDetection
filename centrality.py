import os
import json

import networkx as nx
import pandas as pd


def compute_centrality(db: bool):
    all_dfs = []
    d = "graph" if db else "graph_nodb"
    for f in os.scandir(d):
        with open(f, 'r') as fi:
            g = json.load(fi)
        G = nx.node_link_graph(g, edges="edges", nodes="nodes", name="name", source="sender", target="receiver",
                               multigraph=False, directed=True)
        name = f.name.replace("_gwcc.json", "") if db else f.name.replace("_gwcc_noDB.json", "")
        graph_df = pd.DataFrame(columns=["MS_system", "node"])
        for node in G.nodes:
            graph_df.loc[len(graph_df)] = [name, node]
        norm_degree_centrality = nx.degree_centrality(G)
        degree_centrality = {k: int(v * (len(G.nodes) - 1)) for k, v in norm_degree_centrality.items()}
        norm_in_degree_centrality = nx.in_degree_centrality(G)
        in_degree_centrality = {k: int(v * (len(G.nodes) - 1)) for k, v in norm_in_degree_centrality.items()}
        norm_out_degree_centrality = nx.out_degree_centrality(G)
        out_degree_centrality = {k: int(v * (len(G.nodes) - 1)) for k, v in norm_out_degree_centrality.items()}
        clustering_coefficient = nx.clustering(G)

        graph_df[f"Degree Centrality"] = graph_df["node"].map(degree_centrality)
        graph_df[f"Norm. Degree Centrality"] = graph_df["node"].map(norm_degree_centrality)
        graph_df[f"In-degree Centrality"] = graph_df["node"].map(in_degree_centrality)
        graph_df[f"Norm. In-degree Centrality"] = graph_df["node"].map(norm_in_degree_centrality)
        graph_df[f"Out-degree Centrality"] = graph_df["node"].map(out_degree_centrality)
        graph_df[f"Norm. Out-degree Centrality"] = graph_df["node"].map(norm_out_degree_centrality)
        graph_df[f"Eigenvector Centrality"] = graph_df["node"].map(
            nx.eigenvector_centrality(G, max_iter=1000))
        graph_df[f"Betweenness Centrality"] = graph_df["node"].map(nx.betweenness_centrality(G))
        graph_df[f"Closeness Centrality"] = graph_df["node"].map(nx.closeness_centrality(G))
        graph_df[f"PageRank Centrality"] = graph_df["node"].map(nx.pagerank(G))
        hubs, authorities = nx.hits(G)
        graph_df[f"Hub Score"] = graph_df["node"].map(hubs)
        graph_df[f"Authority Score"] = graph_df["node"].map(authorities)
        G = nx.Graph(G)
        graph_df[f"Subgraph Centrality"] = graph_df["node"].map(nx.subgraph_centrality(G))
        graph_df["Clustering"] = graph_df["node"].map (clustering_coefficient)

        all_dfs.append(graph_df)
    all_dfs = sorted(all_dfs, key=lambda d: d["MS_system"].iloc[0].casefold())
    df_db = pd.concat(all_dfs)
    # df_db = df_db.sort_values(by="Eigenvector Centrality", ascending=False)
    df_db.to_csv(f"metrics_centrality_db_{db}.csv", index=False, header=True)


compute_centrality(True)
compute_centrality(False)
