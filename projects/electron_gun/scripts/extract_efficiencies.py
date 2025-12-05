import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from uncertainties import ufloat as uf


results_file = "projects/electron_gun/results/fit_results.csv"



df = pd.read_csv(results_file)
use_configs = ["pset1", "pset2", "pset3", "pset4"]

df = df[df["config"].isin(use_configs)]

normalized_amplitudes = df["amplitude"] / df["exposure"]

# plot normalized amplitudes for each material, but separate subplots for each material, and different colors for each config
plt.figure()
materials = df["material"].unique()
fit_results = {material: defaultdict(dict) for material in materials}
for material in materials:
    plt.subplot(2, 2, np.where(materials == material)[0][0] + 1)
    mat_mask = df["material"] == material
    mat_amplitudes = normalized_amplitudes[mat_mask]
    mat_currents = df["current_uA"][mat_mask]
    configs = df["config"][mat_mask]
    for config in configs.unique():
        config_mask = configs == config

        mat_con_currents = mat_currents[config_mask]
        mat_con_amplitudes = mat_amplitudes[config_mask]

        # make linear fit to the data
        popt, pcov = curve_fit(lambda x, m, b: m * x + b, mat_con_currents, mat_con_amplitudes)
        m, b = popt
        m_err, b_err = np.sqrt(np.diag(pcov))
        fit_results[material][config]['slope'] = uf(m, m_err)
        fit_results[material][config]['intercept'] = uf(b, b_err)
        x_fit = np.linspace(min(mat_con_currents), max(mat_con_currents), 100)
        y_fit = m * x_fit + b
        plt.scatter(mat_currents[config_mask], mat_amplitudes[config_mask], label=config)
        plt.plot(x_fit, y_fit, label=f"{config} fit")
    
    plt.title(f"Material: {material}")
    plt.xlabel("Current (uA)")
    plt.ylabel("Normalized intensity (a.u.)")
    plt.legend()
    plt.grid()

# compute slopes normalized to specific material
reference_mat = "tube_phosphor"
for material, configs in fit_results.items():
    for config in configs:
        slope = fit_results[material][config]['slope']
        ref_slope = fit_results[reference_mat][config]['slope']
        normalized_slope = slope / ref_slope
        fit_results[material][config]['normalized_slope'] = normalized_slope

print("Fit Results:")
for material, configs in fit_results.items():
    print(f"Material: {material}")
    for config, params in configs.items():
        print(f"  Config: {config}, Slope: {params['slope']:.2u}, Normalized Slope: {params['normalized_slope']:.2u}, Intercept: {params['intercept']:.2u}")

df_results_dict = defaultdict(list)
for material, configs in fit_results.items():
    sums = defaultdict(lambda: uf(0, 0))

    for config, params in configs.items():
        slope = params['slope']
        normalized_slope = params['normalized_slope']
        intercept = params['intercept']
        df_results_dict["material"].append(material)
        df_results_dict["config"].append(config)
        df_results_dict["slope"].append(f"{slope:.2u}")
        df_results_dict["normalized_slope"].append(f"{normalized_slope:.2u}")
        df_results_dict["intercept"].append(f"{intercept:.2u}")
        sums['slope'] += slope
        sums['normalized_slope'] += normalized_slope
        sums['intercept'] += intercept

    df_results_dict["material"].append(material)
    df_results_dict["config"].append("AVERAGE")
    df_results_dict["slope"].append(f"{sums['slope']/len(configs):.2u}")
    df_results_dict["normalized_slope"].append(f"{sums['normalized_slope']/len(configs):.2u}")
    df_results_dict["intercept"].append(f"{sums['intercept']/len(configs):.2u}")

df_results = pd.DataFrame(df_results_dict)
print("\nSummary DataFrame:")
print(df_results)

df_results.to_csv("projects/electron_gun/results/efficiency_summary.csv", index=False)
plt.tight_layout()
plt.show()
    