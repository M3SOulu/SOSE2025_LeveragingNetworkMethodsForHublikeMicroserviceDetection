# Replication package for "Leveraging Network Methods for Hub-like Microservice Detection"

This is the replication package for our paper "Leveraging Network Methods for Hub-like Microservice Detection".

# Contents
This repository contains the following files:

- [Raw](Raw/graph) Raw SDG networks taken from Bakhtin et al. [1]
- [Metrics](Metrics) Network metrics computed from raw data
  - [CentralityMetrics.csv](Metrics/CentralityMetrics.csv) All the computed network centrality metrics
  - [ClusteringHubStrength.csv](Metrics/ClusteringHubStrength.csv) Nodes sorted by hubs strength according to RQ1.3 for each used centrality
- [Results](Results) Results gathered to answer RQ1-3
  - [RQ1](Results/RQ1) Results for RQ1
    - [ScaleFreeTest.csv](Results/RQ1/ScaleFreeTest.csv) Results of testing the degree distributions for Scale-free property (RQ1.1)
    - [Kirkley.json](Results/RQ1/Kirkley.json) Results of using four methods provided by Alec Kirkley [2] for hub detection in networks (RQ1.2)
    - [ClusteringHubsAgreement.csv](Results/RQ1/ClusteringHubsAgreement.csv) Nodes considered hubs by examining the scatter plots of centrality and clustering coefficient for each centrality (RQ1.3)
    - [HubsAll.csv](Results/RQ1/HubsAll.csv) A dataframe of all microservice from all systems labelled as hub or non-hub by all approaches (including Arcan method, RQ1.4)
    - [HubsTrue.csv](Results/RQ1/HubsTrue.csv) A dataframe of all microservices from all systems labelled as hub by at least one approach
    - [HubCounts.csv](Results/RQ1/HubCounts.csv) Count of microservice detect as hub for each approach
  - [RQ2](Results/RQ2) Results for RQ2 (agreement between the methods)
    - [HubJaccard_incoming.csv](Results/RQ2/HubJaccard_incoming.csv) Jaccard coefficient of agreement between the methods analyzing the incoming connections
    - [HubJaccard_outgoing.csv](Results/RQ2/HubJaccard_outgoing.csv) Jaccard coefficient of agreement between the methods analyzing the outgoing connections
    - [HubJaccard_all.csv](Results/RQ2/HubJaccard_all.csv) Jaccard coefficient of agreement between the methods analyzing both the incoming and outgoing connections
  - [RQ3](Results/RQ3) Results for RQ3 (precision through manual validation)
    - [ManualValidation.csv](Results/RQ3/ManualValidation.csv) Nodes detected as hub by at least one approach labelled as True (Hub), Infra (Infrastructural hub), or False (non-Hub)
    - [Precision.csv](Results/RQ3/Precision.csv) Precision of the hub detection approach calculated by considering IH as TP, FP, or ignoring them
- [Scripts](Scripts) Scripts used to perform the study
  - [centrality.py](Scripts/centrality.py) Compute all the centrality metrics and clustering coefficient as well as make the comparison figure of degree and degree centrality, and all SDG figures
  - [scale-free-test.py](Scripts/scale-free-test.py) Test the networks for the scale-free property by fitting a power law to the degree distribution
  - [kirkley.py](Scripts/kirkley.py) Perform hub detection with four approaches provided by Alec Kirkley [2]
  - [clustering.py](Scripts/clustering.py) Compute the hub-like strength by subtracting the clustering coefficient from the centrality metric and make the scatter plots for RQ1.3
  - [merge_results.py](Scripts/merge_results.py) Combine the results of RQ1.1-RQ1.3, generate results for RQ1.4 and agreement and precision for RQ2 and RQ3
- [Figures](Figures) All the figures necessary for the work
  - [SDGs](Figures/SDGs) SDGs of all studied systems
  - [ClusteringScatter](Figures/ClusteringScatter) Scatter plots of centrality and clustering for RQ1.3
  - [Comparison.pdf](Figures/Comparison.pdf) Comparison of CDFs of Degree and Degree centrality
  - [ScaleFree](Figures/ScaleFree.pdf) Results of fitting a power law to the degree distributions
  - [HubJaccard_incoming.pdf](Figures/HubJaccard_incoming.pdf) Jaccard coefficient of agreement between methods analyzing incoming connections
  - [HubJaccard_outgoing.pdf](Figures/HubJaccard_outgoing.pdf) Jaccard coefficient of agreement between methods analyzing outgoing connections
  - [HubJaccard_all.pdf](Figures/HubJaccard_all.pdf) Jaccard coefficient of agreement between methods analyzing both the incoming and outgoing connections

# References

[1] A. Bakhtin, M. Esposito, V. Lenarduzzi and D. Taibi, "Network Centrality as a New Perspective on Microservice Architecture," 2025 IEEE 22nd International Conference on Software Architecture (ICSA), Odense, Denmark, 2025, pp. 72-83, doi: 10.1109/ICSA65012.2025.00017.
[2] Kirkley A. Identifying hubs in directed networks. Physical Review E. 2024 Mar;109(3):034310.
