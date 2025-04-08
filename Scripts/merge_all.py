import json

import pandas as pd

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
understand = pd.read_csv("Metrics/metrics_understand.csv")

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
understand["Microservice"] = understand["MS_system"] + '_' + understand["Microservice"]

centrality = pd.read_csv("Metrics/metrics_centrality.csv")
understand = pd.merge(understand, centrality, on=["MS_system", "Microservice"], how="inner")

understand = understand.sort_values(by=["MS_system", "Microservice"])
understand.to_csv("Metrics/metrics_merged.csv", index=False, header=True)
