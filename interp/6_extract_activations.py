import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append("..")
import argparse
import json
import torch
import lm_eval
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

from model import CustomLlamaConfig, CustomLLaMA
from model_api import CustomModelHandler, format_prompt
from model_api import format_prompt


device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16


def extract_activations(model_path, embedding_type, base_model):
    if embedding_type in {"double_emb", "forward_rot"}:
        AutoConfig.register("custom_llama", CustomLlamaConfig)
        AutoModelForCausalLM.register(CustomLlamaConfig, CustomLLaMA)
    
    if embedding_type == "base":
        system_prompt_len = 30
        template_infix_len = 2
        template_suffix_len = 0 
        embedding_type = "single_emb" # There is not 'base' embedding type in model_api.py this is a hack only to set correct template size.
    elif embedding_type == "single_emb":
        system_prompt_len = 55
        template_infix_len = 7
        template_suffix_len = 5 
    elif embedding_type == "ise":
        system_prompt_len = 55
        template_infix_len = 8
        template_suffix_len = 5 
    elif embedding_type in {"double_emb", "forward_rot"}:
        system_prompt_len = 55
        template_infix_len = 8
        template_suffix_len = 5 
    else:
        raise ValueError(f"Unknown embedding type: {args.embedding_type}")

    print(f"Loding model from {model_path}")
    handler = CustomModelHandler(model_path, base_model, base_model, model_path, None,
        0, embedding_type=embedding_type, load_from_checkpoint=True, model_dtype=dtype
    )

    print(f"Moving the model to {device}")
    handler.model.to(device)


    print(f"Getting the template")
    with open("../data/prompt_templates.json", "r") as f:
        templates = json.load(f)
    template = templates[0]

    # Download from here https://github.com/egozverev/embeddings_for_separation/blob/main/data/train_data/alpaca_adv50percent.json
    alpaca_data_path = "../data/alpaca_adv50percent.json"
    with open(alpaca_data_path, "r") as f:
        alpaca_data = json.load(f)

    dataset_name = "alpaca_adv50percent"

    # set seed 
    torch.manual_seed(42)

    subsample_size = 0.01
    subsample_indices = torch.randperm(len(alpaca_data))[:int(subsample_size*len(alpaca_data))]
    subsampled_data = [alpaca_data[i] for i in subsample_indices]

    print(f"Gathering hidden states for {len(subsampled_data)} examples")
    inst_hidden_all = []
    data_hidden_all = []

    for example in tqdm(subsampled_data):
        instruction_text = example["instruction"]
        data_text = example["input"]

        instruction_prompt = format_prompt(instruction_text, template, "system")
        data_prompt = format_prompt(data_text, template, "user")

        output, inst_tokens, data_tokens, probe_tokens, inst_hidden, data_hidden, probe_hidden, last_hidden, inp = handler.generate_one_token_with_hidden_states(
            instruction_prompt, data_prompt, system_prompt_len=system_prompt_len, template_infix_len=template_infix_len, template_suffix_len=template_suffix_len,
    )
        inst_hidden_all.append(inst_hidden.to("cpu"))
        data_hidden_all.append(data_hidden.to("cpu"))

        torch.cuda.empty_cache()

    inst_hidden_all = torch.cat(inst_hidden_all, dim=0)
    data_hidden_all = torch.cat(data_hidden_all, dim=0)

    short_model_name = model_path.split("/")[-1]
    print(f"Computed short model name for saving: {short_model_name}")

    hidden_states_save_dir = f"../interp/hidden_states_dp/{dataset_name}/{short_model_name}"
    os.makedirs(hidden_states_save_dir, exist_ok=True)
    inst_save_path = f"{hidden_states_save_dir}/inst_hidden_states.pt"
    data_save_path = f"{hidden_states_save_dir}/data_hidden_states.pt"

    print(f"Saving hidden states to {inst_save_path} and {data_save_path}")
    torch.save(inst_hidden_all, inst_save_path)
    torch.save(data_hidden_all, data_save_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract activations from a given model on the adversarial alpaca dataset.")

    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--embedding_type', type=str, required=True, help='Type of embedding used in the model', choices=["base", "single_emb", "double_emb", "forward_rot", "ise"])
    parser.add_argument('--base_model', type=str, default=None, help='Path to the base model')


    args = parser.parse_args()

    extract_activations(args.model, args.embedding_type, args.base_model)

