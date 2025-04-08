# Adapted from https://zenodo.org/records/14748464

import os

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
df = pd.concat(all_dfs)
df.to_csv("Metrics/metrics_understand.csv", index=False, header=True)