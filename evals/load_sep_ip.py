import json
import urllib.request
import os


if __name__ == "__main__":
    original_sep_data_path = "../data/SEP_dataset_1k.json"
    with open(original_sep_data_path, "r") as f:
        sep_data = json.load(f)

    data = []
    for example in sep_data:
        sep_example_in_alpaca_format = {
            "instruction": example["system_prompt_instructed"],
            "input": example["prompt_clean"],
        }
        data.append(sep_example_in_alpaca_format)
    
    save_path = "data/sep_ip/examples.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)