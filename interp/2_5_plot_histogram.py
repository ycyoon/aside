import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


parser = argparse.ArgumentParser(description="Plot inst feature activation histogram for four models.")
parser.add_argument("--base", type=str, required=True, help="Base model name")
parser.add_argument("--single", type=str, required=True, help="Single model name")
parser.add_argument("--ise", type=str, required=True, help="ISE model name")
parser.add_argument("--double", type=str, required=True, help="Double model name")
parser.add_argument("--dataset", type=str, required=True, choices=["full", "injected", "not_injected"], help="Dataset name")
parser.add_argument("--layer", type=int, required=True, help="Layer number to plot")
parser.add_argument("--short-version", action="store_true", help="Only plot two histograms for Vanilla and ASIDE")
args = parser.parse_args()




# Set global style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Nimbus Roman']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42



# Create mapping for model paths and titles
if args.short_version:
    model_paths = [
        args.single,
        args.double,
    ]
    model_titles = ["Vanilla", "ASIDE"]
else:
    model_paths = [
        args.base,
        args.single,
        args.ise,
        args.double,
    ]
    model_titles = ["Base", "Vanilla", "ISE", "ASIDE"]


# Define a function to plot histograms
def plot_histograms(
        ax, inst_activations, data_activations, probe_activations, title, data_with_positive_feature_activation,
        probe_with_positive_feature_activations, show_xlabel=True, show_ylabel=False
    ):
    fixed_min = -6.2
    fixed_max = 6.2
    
    min_data_activation = data_activations.min()
    min_inst_activation = inst_activations.min()
    min_activation = min(min_data_activation, min_inst_activation)

    max_data_activation = data_activations.max()
    max_inst_activation = inst_activations.max()
    max_activation = max(max_data_activation, max_inst_activation)

    absolute_maximum = max(max_activation, abs(min_activation))

    fixed_min = -absolute_maximum
    fixed_max = absolute_maximum

   
    sns.histplot(inst_activations, bins=30, color="darkorchid", alpha=0.6, 
                label="Instruction tokens", ax=ax, kde=True, 
                line_kws={"linewidth": 4}, binrange=(fixed_min, fixed_max), linewidth=0,
                )
    sns.histplot(data_activations, bins=30, color="forestgreen", alpha=0.6, 
                label="Data tokens", ax=ax, kde=True, 
                line_kws={"linewidth": 4}, binrange=(fixed_min, fixed_max), linewidth=0,
                )
 
    sns.histplot(probe_activations, bins=30, color="darkorange", alpha=0.6, 
                 label="Probe tokens", ax=ax, kde=True,
                 line_kws={"linewidth": 4}, binrange=(fixed_min, fixed_max), linewidth=0,
                 )

    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    
    # Add text annotation for percentage of positive activations
    # ax.text(0.6, 5000, f"Positive: {percent:.0%}", 
        #    fontsize=34, color="darkgreen", fontweight='medium',
        #    bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=3))
    
    # Calculate percentage string
    data_positive_str = f"{data_with_positive_feature_activation:.0%}"
    # Add text annotation for percentage of positive activations
    ax.text(0.6, 12500, f"Data: {data_positive_str}", 
        fontsize=34, color="darkgreen", fontweight='medium',
        bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=3))

    probe_positive_str = f"{probe_with_positive_feature_activation:.0%}"
    ax.text(0.6, 8000, f"Probe: {probe_positive_str}",
        fontsize=34, color="darkorange", fontweight='medium',
        bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=3))
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_axisbelow(True)
    
    # Customize spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
       # Labels and title
    if show_xlabel:
        ax.set_xlabel("Concept Activation", fontsize=26)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])  # Remove x-tick labels
    
    if show_ylabel:
        ax.set_ylabel("# of Examples", fontsize=26)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])  # Remove y-tick labels
    
    ax.set_xlim(fixed_min, fixed_max)
    ax.set_ylim(0, 14200)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    
    ax.set_xticks(np.arange(-6.0, 6.2, 3.0))
    ax.set_yticks(np.arange(0, 24001, 4000))
    
    # Customize x and y axis tick font
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    # Add title as text in upper left
    ax.text(fixed_min + 0.4, 9500, title, fontsize=30, fontweight='bold')

# Create a 2x2 grid of subplots
if args.short_version:
    fig = plt.figure(figsize=(16, 4))
    axes = fig.subplots(1, 2)
else:
# Create the main figure
    fig = plt.figure(figsize=(16, 8))
    axes = fig.subplots(2, 2)

# Dataset and layer info
if args.dataset == "full":
    dataset_to_load = "SEP_first1000"
elif args.dataset == "injected":
    dataset_to_load = "SEP_first1000_injected"
elif args.dataset == "not_injected":
    dataset_to_load = "SEP_first1000_not_injected"
else:
    raise ValueError("Invalid dataset name. Choose from 'full', 'injected', or 'not_injected'.")
layer_to_load = args.layer

# For each model, load data and create subplot
for i, (model_path, title) in enumerate(zip(model_paths, model_titles)):
    row = i // 2
    col = i % 2
    
    # Load data for this model
    inst_path = f"../interp/feature_activations_nonorm/{model_path}_{dataset_to_load}_layer{layer_to_load}/probe_middle/inst_activations.npy"
    data_path = f"../interp/feature_activations_nonorm/{model_path}_{dataset_to_load}_layer{layer_to_load}/probe_middle/data_activations.npy"
    probe_path = f"../interp/feature_activations_nonorm/{model_path}_{dataset_to_load}_layer{layer_to_load}/probe_middle/probe_activations.npy"
    
    inst_activations = np.load(inst_path)
    data_activations = np.load(data_path)
    probe_activations = np.load(probe_path)
    
    # Calculate percentage of positive activations
    data_with_positive_feature_activation = (data_activations > 0).sum() / len(data_activations)
    probe_with_positive_feature_activation = (probe_activations > 0).sum() / len(probe_activations)
    
    # Plot on the respective axis
    show_ylabel = (col == 0)  # Show ylabel for the first column
    show_xlabel = True if args.short_version else (row == 1)  # Only show x-axis labels for bottom row

    if args.short_version:
        axes_to_plot = axes[col]
    else:
        axes_to_plot = axes[row, col]
    plot_histograms(axes_to_plot, inst_activations, data_activations, probe_activations, title, 
                   data_with_positive_feature_activation, probe_with_positive_feature_activation, show_xlabel, show_ylabel)
# Create a single legend for the entire figure
handles, labels = axes_to_plot.get_legend_handles_labels()
# reverse the order of the labels
handles = handles[::-1]
labels = labels[::-1]
if args.short_version:
    bbox_to_anchor = (0.5, -0.16)
else:
    bbox_to_anchor = (0.5, 0.04)
leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=bbox_to_anchor, 
               ncol=3, fontsize=28, frameon=False)
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')


# Remove individual legends
for ax in axes.flat:
    if ax.get_legend() is not None:
        ax.get_legend().remove()

plt.tight_layout()
plt.subplots_adjust(bottom=0.26)  # Adjust bottom to make room for the legend
# After creating subplots but before tight_layout
plt.subplots_adjust(hspace=0.15)  # Add this to reduce vertical spacing between rows
figname = f"6_2_combined_{dataset_to_load}.pdf" if not args.short_version else f"6_2_Vanilla_{dataset_to_load}.pdf"
plt.savefig(figname, bbox_inches='tight', dpi=300)
plt.show()