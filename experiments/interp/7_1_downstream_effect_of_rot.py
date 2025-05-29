import argparse
import json
import sys
sys.path.append("..")
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from model import CustomLlamaConfig, CustomLLaMA
from model_api import CustomModelHandler, format_prompt
from model_api import format_prompt



device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16


def plot_cosine_sims(single_path, ise_path, double_path):
    single_short_model_name = single_path.split("/")[-1]
    ise_short_model_name = ise_path.split("/")[-1]
    double_short_model_name = double_path.split("/")[-1]

    single_hidden_states_save_dir = f"../interp/last_acts/SEP_first100/{single_short_model_name}"
    ise_hidden_states_save_dir = f"../interp/last_acts/SEP_first100/{ise_short_model_name}"
    double_hidden_states_save_dir = f"../interp/last_acts/SEP_first100/{double_short_model_name}"
    single_last_save_path = os.path.join(single_hidden_states_save_dir, "last_hidden_states.pt")
    ise_last_save_path = os.path.join(ise_hidden_states_save_dir, "last_hidden_states.pt")
    double_last_save_path = os.path.join(double_hidden_states_save_dir, "last_hidden_states.pt")
    single_last_hidden_all = torch.load(single_last_save_path)
    ise_last_hidden_all = torch.load(ise_last_save_path)
    double_last_hidden_all = torch.load(double_last_save_path)
    print(f"Loaded the data from {single_last_save_path}, {ise_last_save_path} and {double_last_save_path}")


    print(f"Computing mean and std")
    double_per_example_cosine_similarities = torch.nn.functional.cosine_similarity(double_last_hidden_all, single_last_hidden_all, dim=-1)
    ise_per_example_cosine_similarities = torch.nn.functional.cosine_similarity(ise_last_hidden_all, single_last_hidden_all, dim=-1)

    avg_double_per_layer_cosine_similarities = double_per_example_cosine_similarities.mean(dim=0)
    std_double_per_layer_cosine_similarities = double_per_example_cosine_similarities.std(dim=0)

    avg_ise_per_layer_cosine_similarities = ise_per_example_cosine_similarities.mean(dim=0)
    std_ise_per_layer_cosine_similarities = ise_per_example_cosine_similarities.std(dim=0)

    num_layers = avg_double_per_layer_cosine_similarities.shape[0]


    # Set the style
    # plt.style.use('seaborn-paper')
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Nimbus Roman']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42  # Ensure text is editable in PDF


    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color palette that's colorblind-friendly
    colors = ['blue', "orange", "green"]

    palette = sns.color_palette("tab10")
    colors = palette[0:3]


    color1 = "lightseagreen"
    color2 = "darkcyan"

    color3 = "lightcoral"
    color4 = "firebrick"


    # First plot the ASIDE vs Vanilla
    # Before your current plot line, add this:
    ax.fill_between(range(num_layers), 
                    avg_double_per_layer_cosine_similarities - std_double_per_layer_cosine_similarities,
                    avg_double_per_layer_cosine_similarities + std_double_per_layer_cosine_similarities,
                    alpha=0.25,  # Controls transparency
                    color=color1,  # Match the line color
                    linewidth=0)

    # Plot with enhanced styling
    ax.plot(range(num_layers), avg_double_per_layer_cosine_similarities, marker='o', label="ASIDE vs Vanilla",

            color=color1, markersize=10, linewidth=7, markeredgewidth=0.6,
            markeredgecolor='black', markerfacecolor=color2,

            )

    # Now plot the ISE vs Vanilla
    ax.fill_between(range(num_layers),
                    avg_ise_per_layer_cosine_similarities - std_ise_per_layer_cosine_similarities,
                    avg_ise_per_layer_cosine_similarities + std_ise_per_layer_cosine_similarities,
                    alpha=0.25,  # Controls transparency
                    color=color3,  # Match the line color
                    linewidth=0)
    # Plot with enhanced styling
    ax.plot(range(num_layers), avg_ise_per_layer_cosine_similarities, marker='o', label="ISE vs Vanilla",
            color=color3, markersize=10, linewidth=7, markeredgewidth=0.6,
            markeredgecolor='black', markerfacecolor=color4,
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
    ax.set_ylabel('Cosine Similarity', fontsize=26)
    ax.set_ylim(-0.05, 1.05)

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)

    # Customize x and y axis tick font
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=22)

    # Add the last layer to the x-axis
    ax.set_xticks(np.arange(0, num_layers, 4))

    # Add minor ticks
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Legend
    legend = ax.legend(
        frameon=False, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.945),
        ncols=2,
        columnspacing=0.8,
        handletextpad=0.2,
        handlelength=2.0,
        fontsize=25
    )

    # Alternative upper right legend 
    # legend = ax.legend(frameon=True, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=28)
    # legend.get_frame().set_linewidth(1.5)
    # legend.get_frame().set_edgecolor('black')

    # Adjust layout
    plt.tight_layout()
    # plt.show()

    # Save with high DPI
    plt.savefig('6_4_rotation_downstream.pdf', dpi=300, bbox_inches='tight')


def gather_last_activations(single_path, ise_path, double_path, base_model):


    print(f"Loading the data")
    subset_length = 100

    full_sep_path = "../data/SEP_dataset_1k.json"
    with open(full_sep_path, "r") as f:
        full_sep_data = json.load(f)


    data = full_sep_data[:subset_length]
    dataset_name = "SEP_first100"

    print("Loading the template")
    # Get our custom chat template.
    with open("../data/prompt_templates.json", "r") as f:
            templates = json.load(f)
    template = templates[0]



    print(f"Loading the SFT model from {single_path}")
    single_handler = CustomModelHandler(single_path, base_model, base_model, single_path, None,
        0, embedding_type="single_emb", load_from_checkpoint=True, model_dtype=dtype
    )
    single_handler.model.to(device)

    system_prompt_len, template_infix_len, template_suffix_len = single_handler.get_template_parameters(template_type="single")


    print(f"Gathering the hidden states for the SFT model")
    single_last_activations = []

    for example in tqdm(data):
        instruction_text = example["system_prompt_clean"]
        data_text = example["prompt_instructed"]

        instruction_prompt = format_prompt(instruction_text, template, "system")
        data_prompt = format_prompt(data_text, template, "user")

        output, inst_tokens, data_tokens, probe_tokens, inst_hidden, data_hidden, probe_hidden, last_hidden, inp = single_handler.generate_one_token_with_hidden_states(
            instruction_prompt, data_prompt, system_prompt_len=system_prompt_len, template_infix_len=template_infix_len, template_suffix_len=template_suffix_len,
        )

        single_last_activations.append(last_hidden)

    single_last_hidden_all = torch.stack(single_last_activations, dim=0).to("cpu")


    print(f"Saving the SFT data")
    single_short_model_name = single_path.split("/")[-1]
    single_hidden_states_save_dir = f"../interp/last_acts/{dataset_name}/{single_short_model_name}"
    os.makedirs(single_hidden_states_save_dir, exist_ok=True)
    single_last_save_path = os.path.join(single_hidden_states_save_dir, "last_hidden_states.pt")
    torch.save(single_last_hidden_all, single_last_save_path)

    print(f"Deleting the SFT model")
    del single_handler
    torch.cuda.empty_cache()



    print(f"Loading the ISE model from {ise_path}")
    ise_handler = CustomModelHandler(ise_path, base_model, base_model, ise_path, None,
        0, embedding_type="ise", load_from_checkpoint=True, model_dtype=dtype
    )
    ise_handler.model.to(device)

    system_prompt_len, template_infix_len, template_suffix_len = ise_handler.get_template_parameters(template_type="ise")
    print(f"Gathering the hidden states for the ISE model")
    ise_last_activations = []

    for example in tqdm(data):
        instruction_text = example["system_prompt_clean"]
        data_text = example["prompt_instructed"]

        instruction_prompt = format_prompt(instruction_text, template, "system")
        data_prompt = format_prompt(data_text, template, "user")

        output, inst_tokens, data_tokens, probe_tokens, inst_hidden, data_hidden, probe_hidden, last_hidden, inp = ise_handler.generate_one_token_with_hidden_states(
            instruction_prompt, data_prompt, system_prompt_len=system_prompt_len, template_infix_len=template_infix_len, template_suffix_len=template_suffix_len,
        )

        ise_last_activations.append(last_hidden)
    ise_last_hidden_all = torch.stack(ise_last_activations, dim=0).to("cpu")

    print(f"Saving the ISE data")
    ise_short_model_name = ise_path.split("/")[-1]
    ise_hidden_states_save_dir = f"../interp/last_acts/{dataset_name}/{ise_short_model_name}"
    os.makedirs(ise_hidden_states_save_dir, exist_ok=True)
    ise_last_save_path = os.path.join(ise_hidden_states_save_dir, "last_hidden_states.pt")
    torch.save(ise_last_hidden_all, ise_last_save_path)

    print(f"Deleting the ISE model")
    del ise_handler
    torch.cuda.empty_cache()



    print(f"Loading the ASIDE model from {double_path}")
    AutoConfig.register("custom_llama", CustomLlamaConfig)
    AutoModelForCausalLM.register(CustomLlamaConfig, CustomLLaMA)

    double_handler = CustomModelHandler(double_path, base_model, base_model, double_path, None,
        0, embedding_type="forward_rot", load_from_checkpoint=True, model_dtype=dtype
    )
    double_handler.model.to(device)

    system_prompt_len, template_infix_len, template_suffix_len = double_handler.get_template_parameters(template_type="double")

    print(f"Gathering the hidden states for the ASIDE model")
    double_last_activations = []

    for example in tqdm(data):
        instruction_text = example["system_prompt_clean"]
        data_text = example["prompt_instructed"]

        instruction_prompt = format_prompt(instruction_text, template, "system")
        data_prompt = format_prompt(data_text, template, "user")

        output, inst_tokens, data_tokens, probe_tokens, inst_hidden, data_hidden, probe_hidden, last_hidden, inp = double_handler.generate_one_token_with_hidden_states(
            instruction_prompt, data_prompt, system_prompt_len=system_prompt_len, template_infix_len=template_infix_len, template_suffix_len=template_suffix_len,
        )

        double_last_activations.append(last_hidden)
    double_last_hidden_all = torch.stack(double_last_activations, dim=0).to("cpu")
    print(f"Deleting the ASIDE model")
    del double_handler
    torch.cuda.empty_cache()


    print(f"Saving the ASIDE data")
    double_short_model_name = double_path.split("/")[-1]
    double_hidden_states_save_dir = f"../interp/last_acts/{dataset_name}/{double_short_model_name}"
    os.makedirs(double_hidden_states_save_dir, exist_ok=True)
    double_last_save_path = os.path.join(double_hidden_states_save_dir, "last_hidden_states.pt")
    torch.save(double_last_hidden_all, double_last_save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment to see if rotation is unlearned in later layers.")

    parser.add_argument("--single", type=str, required=True, help="Single model name")
    parser.add_argument("--ise", type=str, required=True, help="ISE model name")
    parser.add_argument("--double", type=str, required=True, help="Double model name")
    parser.add_argument('--base_model', type=str, default=None, help='Path to the base model')
    parser.add_argument('--plot', action='store_true', help='If not provided will only gather activations.')


    args =  parser.parse_args()

    if args.plot:
         print(f"Ploting the data gathered before.")
         plot_cosine_sims(args.single, args.ise, args.double)
    else:
        print("Only gathering data, to plot use --plot flag.")
        gather_last_activations(args.single, args.ise, args.double, args.base_model)