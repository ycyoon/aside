import argparse
import sys

if ".." not in sys.path:
    sys.path.append("..")

from transformers import AutoModelForCausalLM, AutoConfig
import torch
import json
from tqdm import tqdm

from model_api import CustomModelHandler, format_prompt
from model import CustomLLaMA, CustomLlamaConfig


device = "cuda" if torch.cuda.is_available() else "cpu"


def run_intervention(model_name, embedding_type, base_model):

    print(f"Loading the model to CPU")
    AutoConfig.register("custom_llama", CustomLlamaConfig)
    AutoModelForCausalLM.register(CustomLlamaConfig, CustomLLaMA)


    handler = CustomModelHandler(model_name, base_model, base_model, model_name, None,
                                    0, embedding_type=embedding_type,
                                    load_from_checkpoint=True,
                                    model_dtype=torch.bfloat16,
                                    )


    system_prompt_len, template_infix_len, template_suffix_len = handler.get_template_parameters(embedding_type)

    print(f"Moving the model to {device}")
    handler.model.to(device)

    with open("../data/prompt_templates.json", "r") as f:
        templates = json.load(f)
    template = templates[0]

    sep_1k_path = "../data/SEP_dataset_1k.json"
    with open(sep_1k_path, "r") as f:
        sep_1k_data = json.load(f)

    clean_injections = 0
    intervened_injections = 0
    for example in tqdm(sep_1k_data):
        system_prompt_clean = example["system_prompt_clean"]
        prompt_clean = example["prompt_clean"]
        prompt_instructed = example["prompt_instructed"]
        witness = example["witness"]
        probe_string = prompt_instructed.replace(prompt_clean, "")

        instruction_prompt = format_prompt(system_prompt_clean, template, "system")
        data_prompt = format_prompt(prompt_instructed, template, "user")

        for intervene_on_probe in [False, True]:
            output, inst_tokens, data_tokens, probe_tokens, inst_hidden, data_hidden, probe_hidden, last_hidden, inp = handler.generate_one_token_with_hidden_states(
                instruction_prompt, data_prompt, system_prompt_len=system_prompt_len, template_infix_len=template_infix_len, template_suffix_len=template_suffix_len,
                max_new_tokens=1024, probe_string=probe_string, intervene_on_probe=intervene_on_probe,
            )
            injection_success = witness.lower() in output.lower()
            if intervene_on_probe:
                intervened_injections += int(injection_success)
            else:
                clean_injections += int(injection_success)

    clean_asr = clean_injections / len(sep_1k_data)
    intervened_asr = intervened_injections / len(sep_1k_data)

    print(f"Clean ASR: {clean_asr}, Count: {clean_injections} / {len(sep_1k_data)}")
    print(f"Intervened ASR: {intervened_asr}, Count: {intervened_injections} / {len(sep_1k_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run intervention experiment on the SEP dataset.")

    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--embedding_type', type=str, required=True, help='Type of embedding used in the model', choices=["double_emb", "forward_rot", "ise"])
    parser.add_argument('--base_model', type=str, default=None, help='Path to the base model')

    args = parser.parse_args()

    run_intervention(args.model, args.embedding_type, args.base_model)