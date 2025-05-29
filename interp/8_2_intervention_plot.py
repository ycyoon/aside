import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


parser = argparse.ArgumentParser(description="Plot clean/asr intervention experiment barplots.")
parser.add_argument("--clean", type=float, required=True, help="Base model name")
parser.add_argument("--intervention", type=float, required=True, help="Single model name")
args = parser.parse_args()

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Nimbus Roman']

# Data
categories = ["Reference", "Intervention"]
values = [args.clean, args.intervention]  # ASR values
# values = [0.145, 0.277]  # ASR values for llama3.1 8b
clean_color = "#1D8B77"  # teal/green
intervention_color = "#CC3311"  # red/orange

# Create plot
fig, ax = plt.subplots(figsize=(4, 3.3))
bars = ax.bar(categories, values, color=[clean_color, intervention_color], alpha=0.7, width=0.8)

# Add labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 0.035,
            f'{height:.1%}', ha='center', va='bottom', fontsize=23, fontweight='bold', color="black")

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.2, axis='y')
ax.set_axisbelow(True)
ax.set_ylim(0, 0.2)
ax.set_ylabel("ASR", fontsize=26)
# ax.set_title("Clean vs. Intervention ASR", fontsize=30, fontweight='bold')
ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.savefig('asr_comparison.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.show()