import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Plot classification accuracies for different models.")
parser.add_argument("--base", type=str, required=True, help="Base model name")
parser.add_argument("--single", type=str, required=True, help="Single model name")
parser.add_argument("--ise", type=str, required=True, help="ISE model name")
parser.add_argument("--double", type=str, required=True, help="Double model name")
args = parser.parse_args()


# Load data
hidden_states_dir = "hidden_states_dp"
dataset_name = "alpaca_adv50percent"
dataset_dir = f"{hidden_states_dir}/{dataset_name}"

# base_model_name = "Llama-3.1-8B"
# single_model_name = "llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix"
# ise_model_name = "llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix"
# double_model_name = "llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix"
base_model_name = args.base
single_model_name = args.single
ise_model_name = args.ise
double_model_name = args.double


base_model_dir = f"{dataset_dir}/{base_model_name}"
single_model_dir = f"{dataset_dir}/{single_model_name}"
ise_model_dir = f"{dataset_dir}/{ise_model_name}"
double_model_dir = f"{dataset_dir}/{double_model_name}"

base_accuracies = json.load(open(f"{base_model_dir}/layer_accuracies.json"))
single_accuracies = json.load(open(f"{single_model_dir}/layer_accuracies.json"))
ise_accuracies = json.load(open(f"{ise_model_dir}/layer_accuracies.json"))
double_accuracies = json.load(open(f"{double_model_dir}/layer_accuracies.json"))

num_layers = len(base_accuracies)

# Create a pandas dataframe with all the accuracies
data = {
    "Layer": list(range(num_layers)),
    "Base": base_accuracies,
    "Vanilla": single_accuracies,
    "ISE": ise_accuracies,
    "ASIDE": double_accuracies
}
df = pd.DataFrame(data)



# Set the style
# plt.style.use('seaborn-paper')
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = ['serif']
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.serif'] = ['Nimbus Roman']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42  # Ensure text is editable in PDF


# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Color palette that's colorblind-friendly
# colors = ['blue', "orange", "green", "magenta"]

palette = sns.color_palette("tab10")
colors = palette[0:4]


# Plot with enhanced styling
ax.plot(range(num_layers), double_accuracies, marker='o', label='ASIDE',
        color=colors[2], markersize=8, linewidth=6, markeredgewidth=0.6, zorder=2,
        markeredgecolor='black'
        )


ax.plot(range(num_layers), single_accuracies, marker='v', label='Vanilla',
        color=colors[1], markersize=8, linewidth=6, markeredgewidth=0.6, zorder=1,
        markeredgecolor='black', 
        )

ax.plot(range(num_layers), ise_accuracies, marker='^', label='ISE',
        color=colors[3], markersize=8, linewidth=6, markeredgewidth=0.6, zorder=1,
        markeredgecolor='black', 
        )

ax.plot(range(num_layers), base_accuracies, marker='s', label='Base', 
        color=colors[0], markersize=8, linewidth=6, markeredgewidth=0.6, zorder=0,
        markeredgecolor='black'
        )

# Customize grid
ax.grid(True, linestyle='--', alpha=0.2)
ax.set_axisbelow(True)

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Labels and title
ax.set_xlabel('Layer', fontsize=26)
ax.set_ylabel('Probe Accuracy', fontsize=26)
ax.set_ylim(0.07, 1.05)

# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
ax.tick_params(axis='both', which='minor', width=1, length=3)

# Customize x and y axis tick font
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Add the last layer to the x-axis
ax.set_xticks(np.arange(0, num_layers, 4))

# Add minor ticks
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

# Legend
legend = ax.legend(frameon=True, loc='lower right', ncols=2,
                 bbox_to_anchor=(0.99, 0.01),
                  fontsize=28)
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_edgecolor('black')

# Adjust layout
plt.tight_layout()

# Save with high DPI
plt.savefig('token_classification.pdf', dpi=300, bbox_inches='tight')
plt.savefig('token_classification.png', dpi=300, bbox_inches='tight')
plt.close()