import json
import urllib.request
import os


if __name__ == "__main__":
    probes_url = "https://raw.githubusercontent.com/egozverev/Should-It-Be-Executed-Or-Processed/refs/heads/main/SEP_dataset/source/probes.json"
    with urllib.request.urlopen(probes_url) as url:
        data = json.load(url)

    for example in data:
        example["input"] = ""
    
    save_path = "data/sep_probes/probes.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)