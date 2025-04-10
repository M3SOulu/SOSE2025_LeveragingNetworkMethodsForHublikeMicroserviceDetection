# Adapted from https://zenodo.org/records/14748464

import os
import json

import pandas as pd


all_dfs = []
for project_dir in os.listdir("Raw/understand"):
    if project_dir == ".DS_Store":
        continue
    project = project_dir.replace("-und", "")
    metrics_path = os.path.join(os.getcwd(), "Raw", "understand", project_dir, f"{project}.csv")
    df = pd.read_csv(metrics_path)
    df = df[df["Kind"] == "Package"]
    df = df.drop(columns=["Kind"])
    df["MS_system"] = project
    cols = ["MS_system"] + [col for col in df.columns if col != "MS_system"]
    df = df[cols]
    all_dfs.append(df)

all_dfs = sorted(all_dfs, key=lambda d: d["MS_system"].iloc[0].casefold())
understand = pd.concat(all_dfs)

with open("Raw/package_map.json", 'r') as f:
    PACKAGE_MAP = json.load(f)

PACKAGES = set(PACKAGE_MAP.keys())
SERVICES = set(PACKAGE_MAP.values())

# Mappping of packages to services
def map_packages(value: str):
    for package, service in PACKAGE_MAP.items():
        if value.startswith(package):
            return service
    return None


# --- Understand metrics

# Remove NaN columns
understand = understand.dropna(axis=1, how='all')

# Map Package to Microservice
understand = understand.rename(columns={"Name": "Package"})
understand["Microservice"] = understand["Package"].map(map_packages)
# Remove rows that are not mapped to a service
understand = understand.dropna(subset=["Microservice"])
understand = understand.drop(columns=["Package"])


# Group by microservice and sum all the metric columns
understand = understand.groupby(["MS_system", 'Microservice']).sum().reset_index()
understand = understand[["MS_system", "Microservice", "CountDeclMethodPublic"]]

centrality = pd.read_csv("Metrics/metrics_centrality.csv")
understand = pd.merge(understand, centrality, on=["MS_system", "Microservice"], how="inner")

understand = understand.sort_values(by=["MS_system", "Microservice"])
understand.to_csv("Metrics/metrics_merged.csv", index=False, header=True)