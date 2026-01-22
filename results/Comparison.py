import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Proposed Method Results
proposed = {
    'theta_12': {
        'RMSE': [0.071692027, 0.071889110, 0.072136328, 0.072062120, 0.067348309,
                 0.072754204, 0.068932988, 0.069602512, 0.069914207, 0.067717329],
        'MPIW': [0.082087338, 0.083103150, 0.083309790, 0.082448957, 0.080178755,
                 0.084884555, 0.081419595, 0.082107107, 0.081782723, 0.080205104],
        'PICP': [0.898, 0.915, 0.899, 0.899, 0.901, 0.913, 0.924, 0.896, 0.892, 0.896]
    },
    'theta_23': {
        'RMSE': [0.026836292, 0.026233522, 0.027530299, 0.025702087, 0.026175519,
                 0.025658791, 0.027672877, 0.026304738, 0.028704279, 0.026914881],
        'MPIW': [0.022316472, 0.022099136, 0.023603582, 0.020776345, 0.022599603,
                 0.021289632, 0.024022095, 0.022706674, 0.024058862, 0.022040865],
        'PICP': [0.894, 0.909, 0.908, 0.907, 0.899, 0.905, 0.897, 0.884, 0.895, 0.896]
    },
    'theta_13': {
        'RMSE': [0.003377414, 0.003707078, 0.003962387, 0.003995770, 0.003820392,
                 0.003819707, 0.003609170, 0.003837313, 0.003792313, 0.003962049],
        'MPIW': [0.003038121, 0.003307599, 0.003258550, 0.003481777, 0.003286879,
                 0.003268390, 0.003599451, 0.003393920, 0.003317111, 0.003447607],
        'PICP': [0.897, 0.902, 0.919, 0.897, 0.900, 0.898, 0.905, 0.907, 0.894, 0.893]
    },
    'delta_cp': {
        'RMSE': [0.870848835, 0.887998164, 0.919465542, 0.850008488, 0.895275235,
                 0.839002013, 0.915739000, 0.885597646, 0.904239476, 0.908747911],
        'MPIW': [1.096895155, 1.115115344, 1.180210432, 1.086133518, 1.114005954,
                 1.065422024, 1.126025464, 1.129577857, 1.117628477, 1.156653353],
        'PICP': [0.892000, 0.897000, 0.911000, 0.895000, 0.913000, 0.885000, 0.904000, 0.906000, 0.902000, 0.912000]
    },
    'm21': {
        'RMSE': [0.000000301, 0.000000291, 0.000000294, 0.000000283, 0.000000285,
                 0.000000299, 0.000000285, 0.000000283, 0.000000292, 0.000000276],
        'MPIW': [0.000000383, 0.000000385, 0.000000383, 0.000000386, 0.000000377,
                 0.000000386, 0.000000376, 0.000000376, 0.000000385, 0.000000374],
        'PICP': [0.889, 0.911, 0.885, 0.909, 0.916, 0.891, 0.895, 0.899, 0.892, 0.902]
    },
    'm31': {
        'RMSE': [0.000000394, 0.000000393, 0.000000393, 0.000000400, 0.000000384,
                 0.000000399, 0.000000375, 0.000000382, 0.000000381, 0.000000384],
        'MPIW': [0.000000566, 0.000000573, 0.000000385, 0.000000556, 0.000000555,
                 0.000000554, 0.000000546, 0.000000565, 0.000000571, 0.000000556],
        'PICP': [0.896, 0.888, 0.904, 0.879, 0.884, 0.891, 0.899, 0.895, 0.906, 0.900]
    }
}

# MCMC Baseline Results
mcmc = {
    'theta_12': {
        'RMSE': [0.12988887, 0.166557053, 0.162755362, 0.135706742, 0.194256431,
                 0.134624999, 0.163958223, 0.146440141, 0.151446388, 0.124608502],
        'MPIW': [0.228072076, 0.285359799, 0.282128788, 0.240323894, 0.322487249,
                 0.243451928, 0.285227714, 0.25360837, 0.262330977, 0.222567905],
        'PICP': [0.887, 0.903, 0.9, 0.865, 0.909, 0.872, 0.9, 0.872, 0.88, 0.865]
    },
    'theta_23': {
        'RMSE': [0.025050616, 0.029125413, 0.024929685, 0.031758451, 0.026938731,
                 0.030419463, 0.027560201, 0.0250908, 0.024460118, 0.027312525],
        'MPIW': [0.05982145, 0.069546017, 0.060422808, 0.074313149, 0.064734813,
                 0.071547233, 0.067152146, 0.060133057, 0.059274691, 0.064013962],
        'PICP': [0.914, 0.913, 0.895, 0.921, 0.903, 0.908, 0.894, 0.888, 0.896, 0.905]
    },
    'theta_13': {
        'RMSE': [0.003383388, 0.003685114, 0.00346566, 0.004088191, 0.003455716,
                 0.003082894, 0.00358826, 0.004779508, 0.00406968, 0.003297413],
        'MPIW': [0.008220776, 0.008899725, 0.00842325, 0.009565445, 0.008471341,
                 0.007570005, 0.008718911, 0.011028615, 0.009624514, 0.008040749],
        'PICP': [0.921, 0.924, 0.899, 0.927, 0.922, 0.896, 0.904, 0.946, 0.922, 0.905]
    },
    'delta_cp': {
        'RMSE': [0.672612514, 0.735072814, 0.711051058, 0.85761139, 0.770084442,
                 0.693034117, 0.768124318, 0.785247169, 1.062194489, 0.741269175],
        'MPIW': [1.633649448, 1.789293264, 1.71971406, 2.012615888, 1.846444984,
                 1.691179218, 1.876074574, 1.876027333, 2.446722027, 1.789111551],
        'PICP': [0.916, 0.899, 0.897, 0.934, 0.914, 0.899, 0.908, 0.914, 0.944, 0.901]
    },
    'm21': {
        'RMSE': [7.4e-07, 7.49e-07, 6.67e-07, 7.35e-07, 7.5e-07,
                 6.66e-07, 6.91e-07, 7.18e-07, 7.89e-07, 7.78e-07],
        'MPIW': [1.672e-06, 1.729e-06, 1.545e-06, 1.69e-06, 1.699e-06,
                 1.54e-06, 1.582e-06, 1.639e-06, 1.791e-06, 1.771e-06],
        'PICP': [0.927, 0.916, 0.909, 0.894, 0.907, 0.899, 0.885, 0.904, 0.9, 0.912]
    },
    'm31': {
        'RMSE': [3.96e-07, 3.99e-07, 3.39e-07, 4.16e-07, 4.03e-07,
                 3.68e-07, 3.81e-07, 3.96e-07, 4.17e-07, 4.55e-07],
        'MPIW': [9.06e-07, 9.27e-07, 8.05e-07, 9.48e-07, 9.19e-07,
                 8.59e-07, 8.78e-07, 9.14e-07, 9.57e-07, 1.039e-06],
        'PICP': [0.928, 0.917, 0.9, 0.902, 0.9, 0.912, 0.892, 0.912, 0.899, 0.917]
    }
}

# Parameter labels
param_labels = {
    'theta_12': r'$\theta_{12}$',
    'theta_23': r'$\theta_{23}$',
    'theta_13': r'$\theta_{13}$',
    'delta_cp': r'$\delta_{\mathrm{CP}}$',
    'm21': r'$\Delta m_{21}^2$',
    'm31': r'$\Delta m_{31}^2$'
}

params = ['theta_12', 'theta_23', 'theta_13', 'delta_cp', 'm21', 'm31']
metrics = ['RMSE', 'MPIW', 'PICP']

# Create figure with subplots (6 rows x 3 columns)
fig, axes = plt.subplots(6, 3, figsize=(12, 16))

# Default matplotlib colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]

for i, param in enumerate(params):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]

        # Prepare data
        data = [proposed[param][metric], mcmc[param][metric]]
        positions = [1, 2]

        # Create boxplot
        bp = ax.boxplot(data, positions=positions, widths=0.6,
                        patch_artist=True,
                        boxprops=dict(linewidth=1.5),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        # Color boxes with default colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Styling
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Proposed', 'MCMC'], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add metric label as column title (only for top row)
        if i == 0:
            ax.set_title(metric, fontsize=12, fontweight='bold', pad=10)

        # Add parameter label as row title (only for leftmost column)
        if j == 0:
            ax.set_ylabel(param_labels[param], fontsize=11, fontweight='bold')

        # Format y-axis for scientific notation if needed
        if param in ['m21', 'm31']:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_fontsize(8)

        # For PICP, set y-axis limits to focus on the relevant range
        if metric == 'PICP':
            all_values = data[0] + data[1]
            y_min = min(all_values) - 0.01
            y_max = max(all_values) + 0.01
            ax.set_ylim([max(0.85, y_min), min(1.0, y_max)])

plt.tight_layout()
plt.savefig('method_comparison_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

for param in params:
    print(f"\n{param_labels[param]}:")
    print("-" * 60)
    for metric in metrics:
        prop_mean = np.mean(proposed[param][metric])
        prop_std = np.std(proposed[param][metric], ddof=1)
        mcmc_mean = np.mean(mcmc[param][metric])
        mcmc_std = np.std(mcmc[param][metric], ddof=1)

        if metric in ['RMSE', 'MPIW']:
            improvement = ((mcmc_mean - prop_mean) / mcmc_mean) * 100
            print(f"  {metric}:")
            print(f"    Proposed: {prop_mean:.6e} ± {prop_std:.6e}")
            print(f"    MCMC:     {mcmc_mean:.6e} ± {mcmc_std:.6e}")
            print(f"    Improvement: {improvement:+.2f}%")
        else:
            print(f"  {metric}:")
            print(f"    Proposed: {prop_mean:.4f} ± {prop_std:.4f}")
            print(f"    MCMC:     {mcmc_mean:.4f} ± {mcmc_std:.4f}")