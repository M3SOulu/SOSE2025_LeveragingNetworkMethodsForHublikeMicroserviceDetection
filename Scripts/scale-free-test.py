import os
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw

# Load data
df = pd.read_csv("Metrics/metrics_centrality.csv")

# Define degree and frequency columns
distributions = [
    "Degree Centrality",
    "In-degree Centrality",
    "Out-degree Centrality",
]

results_summary = []

for centrality in distributions:
    print(f"\n=== Analyzing: {centrality} ===")

    samples = list(df[centrality])
    if len(samples) == 0:
        print("No valid samples extracted.")
        continue

    # Fit power-law model
    fit = powerlaw.Fit(samples, discrete=True, verbose=False,
                       xmin=1)

    # Trigger internal calculations to compute sigma (GoF p-value)
    fit.power_law.pdf()

    # Extract stats
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    ks_stat = fit.power_law.KS()
    pval_gof = fit.power_law.sigma  # Bootstrapped GoF p-value

    print(f"Estimated alpha: {alpha:.4f}")
    print(f"Estimated xmin: {xmin}")
    print(f"KS statistic (GoF): {ks_stat:.4f}")
    print(f"Bootstrapped GoF p-value (sigma): {pval_gof:.4f}")

    # Save result
    results_summary.append({
        "Metric": centrality,
        "alpha": alpha,
        "xmin": xmin,
        "KS_statistic": ks_stat,
        "GoF_p_value": pval_gof
    })

# Save summary to CSV
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(os.path.join("Results", "ScaleFreeTest.csv"), index=False)

deg_scale = df["Degree Centrality"].value_counts(normalize=True).sort_index()
in_deg_scale = df["In-degree Centrality"].value_counts(normalize=True).sort_index()
out_deg_scale = df["Out-degree Centrality"].value_counts(normalize=True).sort_index()

degree = summary_df["alpha"].iloc[0]
in_degree = summary_df["alpha"].iloc[1]
out_degree = summary_df["alpha"].iloc[2]
powerlaw = lambda alpha, x: (alpha - 1) * x ** (-alpha)
degree_power = [powerlaw(degree, x) for x in range(13)]
in_degree_power = [powerlaw(in_degree, x) for x in range(13)]
out_degree_power = [powerlaw(out_degree, x) for x in range(13)]

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.plot(deg_scale.index, deg_scale.values)
plt.plot(degree_power)
plt.title("Degree")
plt.xlim((0, 12))
plt.ylabel("Proportion of nodes, P(k)")
plt.xlabel("Degree, k")
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.plot(in_deg_scale.index, in_deg_scale.values)
plt.plot(in_degree_power)
plt.xlabel("Degree, k")
plt.xlim((0, 12))
plt.ylim((0.0, 0.5))
plt.title("In-degree")
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.plot(out_deg_scale.index, out_deg_scale.values)
plt.plot(out_degree_power)
plt.xlabel("Degree, k")
plt.xlim((0, 12))
plt.ylim((0.0, 0.5))
plt.title("Out-degree")
plt.legend(["P(k) for degree k", "Power law best fit"], loc='right')
plt.tight_layout()

plt.suptitle("Power law fits to determine scale-free property")
plt.tight_layout()
plt.savefig("Figures/ScaleFree.pdf")