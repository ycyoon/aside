import os
import sys
sys.path.append("..")
import argparse
import json
import torch
import lm_eval
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import subprocess
from pathlib import Path

from model import CustomLlamaConfig, CustomLLaMA
from model_api import CustomModelHandler, format_prompt
from model_api import format_prompt

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16


# Computation

def get_probe_activation(residual_stream_vector, probe_weight):
    # return torch.functional.F.cosine_similarity(residual_stream_vector, probe_weight, dim=0).item()
    # Instead compute it as a dot product
    return torch.dot(residual_stream_vector, probe_weight).item()


# Heatmap plot production

def plot_token_similarities(tokens, similarities, n_tokens_per_row=20):
    # Calculate number of rows needed
    n_tokens = len(tokens)
    n_rows = int(np.ceil(len(tokens) / n_tokens_per_row))
    
    # Create figure with appropriate height (1.5 units per row)
    fig, ax = plt.subplots(figsize=(15, 1.5 * n_rows))
    
    # Get actual min and max for better color scaling
    sim_abs_max = max(abs(min(similarities)), abs(max(similarities)))
    sim_min = -sim_abs_max
    sim_max = sim_abs_max
    # sim_min = min(similarities)
    # sim_max = max(similarities)
    
    # Create colormap
    cmap = plt.cm.RdBu_r
    
    # Plot rectangles for each token
    for i, (token, sim) in enumerate(zip(tokens, similarities)):
        # Calculate row and column position
        row = n_rows - 1 - (i // n_tokens_per_row)  # Start from bottom row
        col = i % n_tokens_per_row
        
        # Normalize similarity for color mapping
        norm_sim = (sim - sim_min) / (sim_max - sim_min)
        
        # Create rectangle
        rect = plt.Rectangle((col-0.5, row-0.5), 1, 1, 
                           facecolor=cmap(norm_sim),
                           edgecolor='black',
                           alpha=0.6)
        ax.add_patch(rect)
        
        # Add token text
        ax.text(col, row, token, ha='center', va='center', fontsize=8)
    
    # Set plot limits and remove axes
    ax.set_xlim(-0.5, n_tokens_per_row-0.5)
    ax.set_ylim(-0.5, n_rows-0.5)
    ax.axis('off')
    
    # Add colorbar
    norm = plt.Normalize(sim_min, sim_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Cosine Similarity', orientation='horizontal', 
                ax=ax, fraction=0.1, pad=0.2)
    
    plt.tight_layout()
    return fig


# Latex generation

def _apply_colormap(relevance, cmap):
    
    colormap = cm.get_cmap(cmap)
    return colormap(colors.Normalize(vmin=-1, vmax=1)(relevance))

def generate_latex(words, relevances, cmap="bwr"):

    # Generate LaTeX code
    # latex_code = r'''
    # \documentclass[arwidth=200mm]{standalone} 
    # \usepackage[dvipsnames]{xcolor}
    
    # \begin{document}
    # '''
    latex_code = r'''
    \fbox{
    \parbox{\columnwidth}{
    \setlength\fboxsep{0pt}
    \raggedright  % This helps with better line breaking
    \small
    '''
    for word, relevance in zip(words, relevances):
        rgb = _apply_colormap(relevance, cmap)
        r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)


        if word.startswith('Ġ'):
            word = word.replace('Ġ', ' ')
            latex_code += f'\\hspace{{0.25em}}\\colorbox[RGB]{{{r},{g},{b}}}{{\\strut {word}}}\\allowbreak '
        elif word.startswith('▁'):
            latex_code += f' \\colorbox[RGB]{{{r},{g},{b}}}{{\\strut {word[1:]}}}\\allowbreak '
        elif word == "\n":
            latex_code += "\n"
        else:
            latex_code += f'\\colorbox[RGB]{{{r},{g},{b}}}{{\\strut {word}}}\\allowbreak '

        # if i % 10 == 0:
        #     # add new line every 10 words
        #     latex_code += r'\n'
        # i += 1


    # latex_code += r'}}\end{document}'
    latex_code += r'}}'

    return latex_code


def clean_words(words):
    # add before special characters the escape character /
    if '¨' in words:
        words.remove('¨')
    if 'Ã¨' in words:
        words.remove('Ã¨')
    if 'Ã¨re' in words:
        words.remove('Ã¨re')
    if 'Ã©e' in words:
        words.remove('Ã©e')
    if 'Â' in words:
        words.remove('Â')
    special_characters = ['&', '%', '$', '#', '_', '{', '}']
    llama_special = {"âĢĵ": "-", "Ċ": "\n", "âĢĿ": "\"", "\\": "", "^": ""}
    for i, word in enumerate(words):
        for llama_s in llama_special:
            if llama_s in word:
                words[i] = words[i].replace(llama_s, llama_special[llama_s])
        for special_character in special_characters:
            if special_character in word:
                words[i] = words[i].replace(special_character, '\\' + special_character)
        if 'Ã¨re' in word:
            word = word.replace('Ã¨re', " ")
        if 'Åį' in word:
            word = word.replace('Åį', ' ')
        if "Â" in word:
            word = word.replace("Â", " ")

        if "ł" in word:
            word = word.replace("ł", " ")

    if 'Ã¨re' in words:
        words.remove('Ã¨re')

    if 'Åį' in words:
        words.remove('Åį')

    return words



def plot_qualitative_example(model_path, embedding_type, base_model, layer):
    if embedding_type in {"double_emb", "forward_rot"}:
        AutoConfig.register("custom_llama", CustomLlamaConfig)
        AutoModelForCausalLM.register(CustomLlamaConfig, CustomLLaMA)
    

    print(f"Loding model from {model_path}")
    handler = CustomModelHandler(model_path, base_model, base_model, model_path, None,
        0, embedding_type=embedding_type if embedding_type != "base" else "single_emb", load_from_checkpoint=True, model_dtype=dtype
    )

    system_prompt_len, template_infix_len, template_suffix_len = handler.get_template_parameters(embedding_type)

    print(f"Moving the model to {device}")
    handler.model.to(device)

    print(f"Getting the template")
    with open("../data/prompt_templates.json", "r") as f:
        templates = json.load(f)
    template = templates[0]

    short_model_name = model_path.split("/")[-1]

    # Load the probe
    probe_model_name = short_model_name
    probe_dataset_name = "alpaca_data_cleaned_gpt4"
    probe_path = f"../interp/probe_middle_template/probe_{probe_model_name}_{probe_dataset_name}_layer{layer}_middle.pt"
    print(f"Loading probe from {probe_path}")
    probe = torch.load(probe_path, map_location=device)
    probe_weight = probe["linear.weight"].squeeze() 
    probe_weight = probe_weight.to(dtype)

    # Find the qualitative example, in the SEP dataset
    sep_path = "../data/SEP_dataset_tiny.json"
    with open(sep_path, "r") as f:
        sep_data = json.load(f)

    dataset_name = "SEP_tiny"
    murder = [ex for ex in sep_data if "murder" in ex["witness"]]
    example = murder[2]
    # Some alternatives I have found
    # example = sep_data[23]
    # example = sep_data[29] # Absurdist humor
    example = sep_data[35] # ML, good loking
    # example = sep_data[60] # Joke about news and gym
    instruction_text = example["system_prompt_clean"]
    data_text = example["prompt_instructed"]
    print(f"Instruction: {instruction_text}")
    print(f"Data: {data_text}")

    instruction_prompt = format_prompt(instruction_text, template, "system")
    data_prompt = format_prompt(data_text, template, "user")

    outputs, inps = handler.call_model_api_batch([instruction_prompt], [data_prompt], max_new_tokens=512)
    print(f"Model output: {outputs[0]}")

    output, inst_tokens, data_tokens, probe_tokens, inst_hidden, data_hidden, probe_hidden, last_hidden, inp = handler.generate_one_token_with_hidden_states(
            instruction_prompt, data_prompt, system_prompt_len=system_prompt_len, template_infix_len=template_infix_len, template_suffix_len=template_suffix_len,
    )
    inst_similarities = [get_probe_activation(inst_hidden[i, layer, :], probe_weight) for i in range(len(inst_tokens))]
    print(f"Inst similarities: {inst_similarities}")
    print(f"Positive inst tokens number: {sum([1 for i in inst_similarities if i > 0])}")
    data_similarities = [get_probe_activation(data_hidden[i, layer, :], probe_weight) for i in range(len(data_tokens))]
    print(f"Data similarities: {data_similarities}")
    print(f"Positive data tokens number: {sum([1 for i in data_similarities if i > 0])}")

    clean_inst_tokens = clean_words(inst_tokens)
    clean_data_tokens = clean_words(data_tokens)


    prep_i_sims = np.array(inst_similarities)
    prep_i_sims = prep_i_sims / np.abs(prep_i_sims).max()

    prep_d_sims = np.array(data_similarities)
    prep_d_sims = prep_d_sims / np.abs(prep_d_sims).max()

    inst_latex_code = generate_latex(clean_inst_tokens, prep_i_sims, cmap="bwr")
    print(f"Inst latex code:\n{inst_latex_code}")

    data_latex_code = generate_latex(clean_data_tokens, prep_d_sims, cmap="bwr")
    print(f"Data latex code:\n{data_latex_code}")

    # Now generating the heatmaps for visualization, so one does not have to always paste latex into overleaf
    heatmap_dir = f"../interp/heatmaps/{short_model_name}"
    print(f"Saving heatmaps to {heatmap_dir}")
    Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
    inst_fig = plot_token_similarities(clean_inst_tokens, prep_i_sims)
    plt.savefig(f"{heatmap_dir}/inst_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(inst_fig)
    data_fig = plot_token_similarities(clean_data_tokens, prep_d_sims)
    plt.savefig(f"{heatmap_dir}/data_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(data_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot instruction feature activations for a given model on some qualitative example.")

    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--embedding_type', type=str, required=True, help='Type of embedding used in the model', choices=["base", "single_emb", "double_emb", "forward_rot", "ise"])
    parser.add_argument('--base_model', type=str, default=None, help='Path to the base model')
    parser.add_argument('--layer', type=int, default=15, help='Layer which we used to compute instruction feature')


    args =  parser.parse_args()

    plot_qualitative_example(args.model, args.embedding_type, args.base_model, args.layer)
