import os
import json

import networkx as nx
import pandas as pd


def compute_centrality():
    all_dfs = []
    for f in os.scandir("Raw"):
        with open(f, 'r') as fi:
            g = json.load(fi)
        G = nx.node_link_graph(g, edges="edges", nodes="nodes", name="name", source="sender", target="receiver",
                               multigraph=False, directed=True)
        G.remove_nodes_from(["user"])
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

        graph_df[f"Degree Centrality"] = graph_df["Microservice"].map(degree_centrality)
        graph_df[f"Norm. Degree Centrality"] = graph_df["Microservice"].map(norm_degree_centrality)
        graph_df[f"In-degree Centrality"] = graph_df["Microservice"].map(in_degree_centrality)
        graph_df[f"Norm. In-degree Centrality"] = graph_df["Microservice"].map(norm_in_degree_centrality)
        graph_df[f"Out-degree Centrality"] = graph_df["Microservice"].map(out_degree_centrality)
        graph_df[f"Norm. Out-degree Centrality"] = graph_df["Microservice"].map(norm_out_degree_centrality)
        graph_df[f"Eigenvector Centrality"] = graph_df["Microservice"].map(
            nx.eigenvector_centrality(G, max_iter=1000))
        graph_df[f"Betweenness Centrality"] = graph_df["Microservice"].map(nx.betweenness_centrality(G))
        graph_df[f"Closeness Centrality"] = graph_df["Microservice"].map(nx.closeness_centrality(G))
        graph_df[f"PageRank Centrality"] = graph_df["Microservice"].map(nx.pagerank(G))
        hubs, authorities = nx.hits(G)
        graph_df[f"Hub Score"] = graph_df["Microservice"].map(hubs)
        graph_df[f"Authority Score"] = graph_df["Microservice"].map(authorities)

        G = nx.Graph(G)
        graph_df[f"Subgraph Centrality"] = graph_df["Microservice"].map(nx.subgraph_centrality(G))
        graph_df["Clustering Coefficient"] = graph_df["Microservice"].map(clustering_coefficient)

        all_dfs.append(graph_df)

    all_dfs = sorted(all_dfs, key=lambda d: d["MS_system"].iloc[0].casefold())
    df_db = pd.concat(all_dfs)
    # df_db = df_db.sort_values(by="Eigenvector Centrality", ascending=False)
    df_db['Microservice'] = df_db['MS_system'] + '_' + df_db['Microservice']
    df_db.to_csv(os.path.join("Metrics", "metrics_centrality.csv"), index=False, header=True)

    # Columns you want to calculate fractions for
    columns = ["Degree Centrality", "In-degree Centrality", "Out-degree Centrality"]

    # Get value fractions for each column
    fractions_dict = {col: df_db[col].value_counts(normalize=True) for col in columns}

    # Convert to DataFrames (optional, if you want a tidy format)
    proportions = pd.DataFrame(fractions_dict).fillna(0).reset_index()
    proportions.rename(columns={'index': 'degree'}, inplace=True)

    proportions.to_csv(os.path.join("Metrics", "scale_free_proportion.csv"), index=False, header=True)

    # Create a new DataFrame where each column is the name sorted by that metric
    sorted_names_by_metric = {
        col: df_db.sort_values(by=col, ascending=False)["Microservice"].reset_index(drop=True)
        for col in df_db.columns if col not in ["MS_system", "Microservice", "Clustering Coefficient"]
    }
    sorted_names_by_metric["Clustering"] =  df_db.sort_values(by="Clustering Coefficient", ascending=True)["Microservice"].reset_index(drop=True)
    sorted_names_by_metric = pd.DataFrame(sorted_names_by_metric)

    sorted_names_by_metric.to_csv(os.path.join("Metrics", "sorted_nodes.csv"), index=False)


if __name__ == "__main__":
    compute_centrality()
