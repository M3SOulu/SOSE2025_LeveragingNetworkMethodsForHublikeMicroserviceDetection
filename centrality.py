import os
import json

import networkx as nx
import pandas as pd

all_dfs = []
for f in os.scandir("graph"):
    db = f.name.endswith("_gwcc.json")
    with open(f, 'r') as fi:
        g = json.load(fi)
    G = nx.node_link_graph(g, edges="edges", nodes="nodes", name="name", source="sender", target="receiver",
                           multigraph=False, directed=True)
    name = f.name.replace("_gwcc.json", "") if db else f.name.replace("_gwcc_noDB.json", "")
    graph_df = pd.DataFrame(columns=["MS_system", "node", "DB"])
    for node in G.nodes:
        graph_df.loc[len(graph_df)] = [name, node, db]
    norm_degree_centrality = nx.degree_centrality(G)
    degree_centrality = {k: int(v*(len(G.nodes) - 1)) for k, v in norm_degree_centrality.items()}
    norm_in_degree_centrality = nx.in_degree_centrality(G)
    in_degree_centrality = {k: int(v*(len(G.nodes) - 1)) for k, v in norm_in_degree_centrality.items()}
    norm_out_degree_centrality = nx.out_degree_centrality(G)
    out_degree_centrality = {k: int(v*(len(G.nodes) - 1)) for k, v in norm_out_degree_centrality.items()}

    graph_df["Degree Centrality"] = graph_df["node"].map(degree_centrality)
    graph_df["Norm. Degree Centrality"] = graph_df["node"].map(norm_degree_centrality)
    graph_df["In-degree Centrality"] = graph_df["node"].map(in_degree_centrality)
    graph_df["Norm. In-degree Centrality"] = graph_df["node"].map(norm_in_degree_centrality)
    graph_df["Out-degree Centrality"] = graph_df["node"].map(out_degree_centrality)
    graph_df["Norm. Out-degree Centrality"] = graph_df["node"].map(norm_out_degree_centrality)
    graph_df["Eigenvector Centrality"] = graph_df["node"].map(nx.eigenvector_centrality(G, max_iter=1000))
    graph_df["Betweenness Centrality"] = graph_df["node"].map(nx.betweenness_centrality(G))

    all_dfs.append(graph_df)

all_dfs = sorted(all_dfs, key=lambda d: d["MS_system"].iloc[0].casefold())
df = pd.concat(all_dfs)
df.to_csv("metrics_centrality.csv", index=False, header=True)
