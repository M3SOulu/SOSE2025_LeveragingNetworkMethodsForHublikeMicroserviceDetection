import os
import json

import networkx as nx
import pandas as pd

all_dfs_db = []
all_dfs_nodb = []
for f in os.scandir("graph"):
    db = f.name.endswith("_gwcc.json")
    with open(f, 'r') as fi:
        g = json.load(fi)
    G = nx.node_link_graph(g, edges="edges", nodes="nodes", name="name", source="sender", target="receiver",
                           multigraph=False, directed=True)
    name = f.name.replace("_gwcc.json", "") if db else f.name.replace("_gwcc_noDB.json", "")
    graph_df = pd.DataFrame(columns=["MS_system", "node"])
    for node in G.nodes:
        graph_df.loc[len(graph_df)] = [name, node]
    norm_degree_centrality = nx.degree_centrality(G)
    degree_centrality = {k: int(v*(len(G.nodes) - 1)) for k, v in norm_degree_centrality.items()}
    norm_in_degree_centrality = nx.in_degree_centrality(G)
    in_degree_centrality = {k: int(v*(len(G.nodes) - 1)) for k, v in norm_in_degree_centrality.items()}
    norm_out_degree_centrality = nx.out_degree_centrality(G)
    out_degree_centrality = {k: int(v*(len(G.nodes) - 1)) for k, v in norm_out_degree_centrality.items()}

    graph_df[f"Degree Centrality (DB={db})"] = graph_df["node"].map(degree_centrality)
    graph_df[f"Norm. Degree Centrality (DB={db})"] = graph_df["node"].map(norm_degree_centrality)
    graph_df[f"In-degree Centrality (DB={db})"] = graph_df["node"].map(in_degree_centrality)
    graph_df[f"Norm. In-degree Centrality (DB={db})"] = graph_df["node"].map(norm_in_degree_centrality)
    graph_df[f"Out-degree Centrality (DB={db})"] = graph_df["node"].map(out_degree_centrality)
    graph_df[f"Norm. Out-degree Centrality (DB={db})"] = graph_df["node"].map(norm_out_degree_centrality)
    graph_df[f"Eigenvector Centrality (DB={db})"] = graph_df["node"].map(nx.eigenvector_centrality(G, max_iter=1000))
    graph_df[f"Betweenness Centrality (DB={db})"] = graph_df["node"].map(nx.betweenness_centrality(G))

    if db:
        all_dfs_db.append(graph_df)
    else:
        all_dfs_nodb.append(graph_df)

all_dfs_db = sorted(all_dfs_db, key=lambda d: d["MS_system"].iloc[0].casefold())
all_dfs_nodb = sorted(all_dfs_nodb, key=lambda d: d["MS_system"].iloc[0].casefold())
df_db = pd.concat(all_dfs_db)
df_nodb = pd.concat(all_dfs_nodb)
df = pd.merge(df_db, df_nodb, on=["MS_system", "node"], how="left")
df.to_csv("metrics_centrality.csv", index=False, header=True)
