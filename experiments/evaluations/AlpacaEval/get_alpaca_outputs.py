"""
ASIDE AlpacaEval Output Generation Script

This script generates model outputs for AlpacaEval supporting the ASIDE (Architecturally
 Separated Instruction-Data Embeddings) models and baseline approaches.

The script is designed to evaluate instruction-following capabilities.

Key Features:
- Support for multiple embedding types (vanilla, ISE, ASIDE)
- Batch processing for efficiency
- Configurable prompt templates
- Support for major model families (Llama, Qwen, Mistral, Gemma)

Usage:
    python get_alpaca_outputs.py --data-path data/alpaca_eval.json \
                                --model models/llama_3.1_8b/forward_rot/... \
                                --embedding-type forward_rot \
                                --use-input True \
                                --batch-size 32

"""
import sys
import os

if "../.." not in sys.path:
    sys.path.append("../..")

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig

from model_api import CustomModelHandler, format_prompt
from model import CustomLLaMA, CustomLlamaConfig

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_results_batch(handler, use_input, remove_tags, max_new_tokens, template, examples, batch_size):
    """
    Generate model outputs in batches for a list of instruction-input examples.
    
    Args:
        handler (CustomModelHandler): Initialized model handler with loaded model
        use_input (bool):
                         - True for ASIDE/ISE models (uses instruction-data separation)
                         - False for vanilla models (instruction only)
        remove_tags (bool): Whether to remove response tags from model output
        max_new_tokens (int): Maximum number of tokens to generate per example
        template (dict): Prompt template configuration for formatting
        examples (list): List of dictionaries with 'instruction' and 'input' keys
        batch_size (int): Number of examples to process in each batch
        
    Returns:
        list: List of result dictionaries with keys:
            - 'instruction': The full formatted prompt sent to model
            - 'output': The generated model response  
            - 'generator': Path to the model checkpoint used
            
    Note:
        The function applies different prompt formatting based on embedding type:
        - For ASIDE/ISE: Instructions and data are processed separately
        - For vanilla: Combined instruction+input processing
    """
    responses = []
    instructions = []
    for start_idx in tqdm(range(0, len(examples), batch_size)):
        end_idx = min(start_idx + batch_size, len(examples))
        batch_data = examples[start_idx:end_idx]

        batch_instructions = []

        inst_list = []
        data_list = []
        empty_data = ""
        for example in batch_data:
            instruction = example["instruction"]
            data = example["input"] if use_input else empty_data

            # special for 
            instruction = f"{example['instruction']}\n"
            data = f"{example['input']}\n" if use_input else ""


            batch_instructions.append(instruction + data)


            instruction_text = format_prompt(instruction, template, "system")
            data_text = format_prompt(empty_data, template, "user")
            inst_list.append(instruction_text)
            data_list.append(data_text)


        batch_responses, batch_inps = handler.call_model_api_batch(
            inst_list, data_list, max_new_tokens=max_new_tokens,
        ) 

        responses.extend(batch_responses)
        instructions.extend(batch_instructions)
    
    results = [] 
    # Results should contain keys "instruction" with the full model prompt and "output" with the model output
    for i in range(len(responses)):
        instruction = instructions[i]
        response = responses[i]

        if remove_tags:
            start_tag = "Response: "
            # end_tag = " End of Response."
            if response.startswith(start_tag):
                response = response[len(start_tag):]
            # if response.endswith(end_tag):
            #     response = response[:-len(end_tag)]

        result = {"instruction": instruction, "output": response, "generator": handler.checkpoint_path}
        results.append(result)

    return results


SINGLE_EMB_PATHS = {
    "llama_3.1_8b": {
        "original":       "meta-llama/Llama-3.1-8B",
        "original_inst":  "meta-llama/Llama-3.1-8B-Instruct",
    },
    
    "llama_2_7b": {
        "original":       "meta-llama/Llama-2-7b-hf",
        "original_inst":  "meta-llama/Llama-2-7b-chat-hf",
    },
    "llama_2_13b": {
        "original":       "meta-llama/Llama-2-13b-hf",
        "original_inst":  "meta-llama/Llama-2-13b-chat-hf",
    },
    "Qwen2.5-7B": {
        "original": "Qwen/Qwen2.5-7B",
        "original_inst": "Qwen/Qwen2.5-7B-Instruct",
        "base":  "Qwen/Qwen2.5-7B",
    },
    "Mistral-7B-v0.3": {
        "original": "mistralai/Mistral-7B-v0.3",
        "original_inst": "mistralai/Mistral-7B-Instruct-v0.3",
        "base": "mistralai/Mistral-7B-v0.3",
    }
}

def get_model_outputs(data_path, data_size, use_input, remove_tags, model_name, max_new_tokens, embedding_type, base_model, batch_size, save_dir, seed, tokenizer=None):
    """
    Main function to generate model outputs for AlpacaEval
    
    This function loads a dataset, initializes a model with specified embedding type,
    generates responses, and saves results for downstream evaluation (e.g., AlpacaEval).
    
    Args:
        data_path (str): Path to JSON file containing evaluation examples
        data_size (int): Number of examples to evaluate (-1 for all)
        use_input (bool): Whether to use input field from examples
                         True for ASIDE/ISE models, False for vanilla
        remove_tags (bool): Whether to clean response formatting tags
        model_name (str): Path to model directory or HuggingFace model name
        max_new_tokens (int): Maximum tokens to generate per response
        embedding_type (str): Type of embedding strategy:
                             - 'single_emb': Vanilla single embedding
                             - 'double_emb': Legacy double embedding  
                             - 'forward_rot': ASIDE method with rotation
                             - 'ise': ISE baseline method
        base_model (str): Base model path (used for some embedding types)
        batch_size (int): Number of examples to process per batch
        save_dir (str): Directory to save output JSON file
        seed (int): Random seed for reproducible data sampling
        tokenizer (str, optional): Override tokenizer path
        
    Workflow:
        1. Load and optionally subsample evaluation data
        2. Initialize model with appropriate embedding configuration
        3. Generate responses using batch processing
        4. Save results in AlpacaEval-compatible format
        
    Output:
        Saves a JSON file with format:
        [
            {
                "instruction": "formatted prompt",
                "output": "model response", 
                "generator": "model_path"
            },
            ...
        ]
    """
    print(f"Loading and randomly subsetting the data to size {data_size}")
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # To take a random subset of a list in python we do 
    if data_size != -1:
        random.seed(seed)
        data = random.sample(data, data_size)

    print(f"Loading the model to CPU")
    if embedding_type == "double_emb":
        AutoConfig.register("custom_llama", CustomLlamaConfig)
        AutoModelForCausalLM.register(CustomLlamaConfig, CustomLLaMA)

    # fixed tokenizer to instruct
    try:
        tokenizer = SINGLE_EMB_PATHS.get(model_name.split('/')[-1]).get('original_inst')
    except:
        tokenizer = model_name
    print(f"Tokenzier {tokenizer}")    

    handler = CustomModelHandler(model_name, base_model, base_model, tokenizer, None,
                                0, embedding_type=embedding_type,
                                load_from_checkpoint=True,
                                )
    print(f"Moving the model to {device}")
    handler.model.to(device)
    handler.model.config.use_cache = True

    with open("../../data/prompt_templates.json", "r") as f:
        templates = json.load(f)
    template = templates[0]

    print(f"Generating responses")
    results = generate_results_batch(handler, use_input, remove_tags, max_new_tokens, template, data, batch_size)

    # Get parent dir of data_path
    #short_model_name = model_name.split("/")[-1]
    model_name = model_name.split("models/")[1]
    short_model_name = model_name.replace("/", "_")
    save_filename = f"{short_model_name}_l{data_size}_s{seed}.json"
    save_path = os.path.join(save_dir, save_filename)
    # Make sure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving results to {save_path}")
    with open(save_path, "w+") as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to save model outputs on an alpaca-type dataset')

    # --model "models/Embeddings-Collab/llama_3.1_8b_double_emb_SFTv19_run_7"
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data')
    parser.add_argument('--data-size', type=int, default=-1, help='Size of the data to use')
    parser.add_argument('--use-input', type=bool, default=False, help='Whether to use the input data')
    parser.add_argument('--remove-tags', type=bool, default=True, help='Whether to remove tags from the input data')

    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--max-new-tokens', type=int, default=1024, help='How many new tokens to generate')
    parser.add_argument('--embedding-type', type=str, required=True, help='Type of embedding used in the model', choices=['single_emb', 'double_emb', "forward_rot", "ise"])
    parser.add_argument('--base-model', type=str, default="none", help='Path to the base model')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for the model')

    parser.add_argument('--save-dir', type=str, default="", help='Directory to save the model outputs')


    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')

    args = parser.parse_args()

    if args.save_dir == "":
        args.save_dir = os.path.dirname(args.data_path)

    get_model_outputs(
        args.data_path, args.data_size, args.use_input, args.remove_tags, args.model, args.max_new_tokens, args.embedding_type, args.base_model, args.batch_size, args.save_dir, args.seed
    )

