import os
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw

# Create output directory
output_dir = "ScaleFreeResults"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("Metrics/scale_free_proportion.csv")

# Define degree and frequency columns
distributions = {
    "Degree Centrality": "degree",
    "In-degree Centrality": "degree",
    "Out-degree Centrality": "degree"
}

# Prepare list for summary results
results_summary = []

for label, deg_col in distributions.items():
    print(f"\n=== Analyzing: {label} ===")

    # Expand degree distribution into raw data points
    deg_vals = df[deg_col].astype(int)
    freq_vals = df[label]

    # Create raw samples by repeating degrees according to their (rounded) frequency * 1000
    samples = []
    for d, f in zip(deg_vals, freq_vals):
        samples.extend([d] * int(round(f * 1000)))

    if len(samples) == 0:
        print("No valid samples extracted.")
        continue

    # Fit power-law model
    fit = powerlaw.Fit(samples, discrete=True, verbose=False)

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
        "Metric": label,
        "alpha": alpha,
        "xmin": xmin,
        "KS_statistic": ks_stat,
        "GoF_p_value": pval_gof
    })

    # Plot PDF
    plt.figure(figsize=(8, 6))
    fit.plot_pdf(label='Empirical', color='blue')
    fit.power_law.plot_pdf(label=f'Power Law fit (α={alpha:.2f})', color='red')
    plt.title(f"PDF - {label}")
    plt.xlabel("Degree")
    plt.ylabel("P(k)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pdf_{label.replace(' ', '_').lower()}.png"))
    plt.close()

    # Plot CCDF
    plt.figure(figsize=(8, 6))
    fit.plot_ccdf(label='Empirical', color='blue')
    fit.power_law.plot_ccdf(label=f'Power Law CCDF (α={alpha:.2f})', color='red')
    plt.title(f"CCDF - {label}")
    plt.xlabel("Degree")
    plt.ylabel("P(X ≥ k)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ccdf_{label.replace(' ', '_').lower()}.png"))
    plt.close()

# Save summary to CSV
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(os.path.join(output_dir, "scale_free_summary.csv"), index=False)