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

    # Plot PDF
    plt.figure(figsize=(8, 6))
    fit.plot_pdf(label='Empirical', color='blue')
    fit.power_law.plot_pdf(label=f'Power Law fit (α={alpha:.2f})', color='red')
    plt.title(f"PDF - {centrality}")
    plt.xlabel("Degree")
    plt.ylabel("P(k)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("Results", "ScaleFreeTest", f"pdf_{centrality.replace(' ', '_').lower()}.pdf"))
    plt.close()

    # Plot CCDF
    plt.figure(figsize=(8, 6))
    fit.plot_ccdf(label='Empirical', color='blue')
    fit.power_law.plot_ccdf(label=f'Power Law CCDF (α={alpha:.2f})', color='red')
    plt.title(f"CCDF - {centrality}")
    plt.xlabel("Degree")
    plt.ylabel("P(X ≥ k)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("Results", "ScaleFreeTest", f"ccdf_{centrality.replace(' ', '_').lower()}.pdf"))
    plt.close()

# Save summary to CSV
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(os.path.join("Results", "ScaleFreeTest", "scale_free_summary.csv"), index=False)