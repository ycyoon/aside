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


def gather_feature_activation_data(model_path, embedding_type, base_model, layer, full_data_only):
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

    # Load the dataset
    subset_length = 1000
    full_sep_path = "../data/SEP_dataset_1k.json"
    with open(full_sep_path, "r") as f:
        full_sep_data = json.load(f)


    subset_data = full_sep_data[:subset_length]
    dataset_name = f"SEP_first{subset_length}"


    if full_data_only:
        dataset_and_name = [
            (subset_data, dataset_name),
        ]
    else:
        # Take model outputs in order to later compute subsets where injection happened or not
        # Generate model outputs in batches of 8 and save them for future loading
        batch_size = 16
        max_new_tokens = 512
        all_outputs = []

        for i in tqdm(range(0, len(subset_data), batch_size)):
            batch_data = subset_data[i:i+batch_size]
            instruction_prompts = [format_prompt(ex["system_prompt_clean"], template, "system") for ex in batch_data]
            data_prompts = [format_prompt(ex["prompt_instructed"], template, "user") for ex in batch_data]
            outputs, inps = handler.call_model_api_batch(instruction_prompts, data_prompts, max_new_tokens=max_new_tokens)
            all_outputs.extend(outputs)

        # Save the outputs to a json file
        outputs_path = f"../interp/cached_outputs/{short_model_name}_{dataset_name}_outputs.json"
        os.makedirs(os.path.dirname(outputs_path), exist_ok=True)
        with open(outputs_path, "w") as f:
            json.dump(all_outputs, f, indent=4)


        # Split full_sep_data into subsets where the injection was successful or not
        sep_injected = []
        sep_not_injected = []

        for example, output in tqdm(zip(subset_data, all_outputs)):
            witness = example["witness"]

            if witness.lower() in output.lower():
                sep_injected.append(example)
            else:
                sep_not_injected.append(example)

        print(f"Injected: {len(sep_injected)} out of {len(subset_data)}, {len(sep_injected)/len(subset_data):.1%}")
        print(f"Not injected: {len(sep_not_injected)} out of {len(subset_data)}, {len(sep_not_injected)/len(subset_data):.1%}")

        dataset_and_name = [
            (subset_data, dataset_name),
            (sep_injected, f"{dataset_name}_injected"),
            (sep_not_injected, f"{dataset_name}_not_injected"),
        ] 

    for dataset, name in dataset_and_name:
        print(f"Processing {name} dataset")

        avg_inst_activations = []
        avg_data_activations = []
        avg_probe_activations = []

        i = 0
        for example in tqdm(dataset):
            instruction_text = example["system_prompt_clean"]
            prompt_instructed = example["prompt_instructed"]
            prompt_clean = example["prompt_clean"]
            probe_string = prompt_instructed.replace(prompt_clean, "")
            data_text = prompt_instructed


            instruction_prompt = format_prompt(instruction_text, template, "system")
            data_prompt = format_prompt(data_text, template, "user")

            output, inst_tokens, data_tokens, probe_tokens, inst_hidden, data_hidden, probe_hidden, last_hidden, inp = handler.generate_one_token_with_hidden_states(
                    instruction_prompt, data_prompt, system_prompt_len=system_prompt_len, template_infix_len=template_infix_len, template_suffix_len=template_suffix_len, probe_string=probe_string,
            )

            inst_similarities = [get_probe_activation(inst_hidden[i, layer, :], probe_weight) for i in range(len(inst_tokens))]
            data_similarities = [get_probe_activation(data_hidden[i, layer, :], probe_weight) for i in range(len(data_tokens))]
            probe_similarities = [get_probe_activation(probe_hidden[i, layer, :], probe_weight) for i in range(len(probe_tokens))]

            avg_inst_activations.extend(inst_similarities)
            avg_data_activations.extend(data_similarities)
            avg_probe_activations.extend(probe_similarities)


        avg_inst_activations = np.array(avg_inst_activations)
        avg_data_activations = np.array(avg_data_activations)
        avg_probe_activations = np.array(avg_probe_activations)
        save_dir = f"../interp/feature_activations_nonorm/{short_model_name}_{name}_layer{layer}/probe_middle/"
        os.makedirs(save_dir, exist_ok=True)

        inst_path = os.path.join(save_dir, "inst_activations.npy")
        data_path = os.path.join(save_dir, "data_activations.npy")
        probe_path = os.path.join(save_dir, "probe_activations.npy")

        np.save(inst_path, avg_inst_activations)
        np.save(data_path, avg_data_activations)
        np.save(probe_path, avg_probe_activations)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather data of instruction feature activation on tokens of the sep dataset with probe in data..")

    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--embedding_type', type=str, required=True, help='Type of embedding used in the model', choices=["base", "single_emb", "double_emb", "forward_rot", "ise"])
    parser.add_argument('--base_model', type=str, default=None, help='Path to the base model')
    parser.add_argument('--layer', type=int, default=15, help='Layer which we used to compute instruction feature')
    parser.add_argument('--full_data_only', action="store_true", help="Only compute for the full dataset.")


    args =  parser.parse_args()

    gather_feature_activation_data(args.model, args.embedding_type, args.base_model, args.layer, args.full_data_only)