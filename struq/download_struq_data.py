import argparse
import datasets
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to save model outputs on an alpaca-type dataset')
    parser.add_argument('--dataset', type=str, default="tatsu-lab/alpaca_farm", help='Name of the dataset on huggingface')
    parser.add_argument('--subset', type=str, default="alpaca_farm_evaluation", help='Name of subset')
    parser.add_argument('--split', type=str, default="eval", help='Split of the dataset to use')
    parser.add_argument('--save-dir', type=str, default="data", help='Directory to save the results')

    args = parser.parse_args()


    dataset = datasets.load_dataset(args.dataset, args.subset)

    # Ensure the save directory exists
    full_save_dir = f"{args.save_dir}/{args.dataset}"
    os.makedirs(full_save_dir, exist_ok=True)
    save_path = f"{full_save_dir}/{args.split}.json"

    # Save the dataset as a json file
    split_data = dataset[args.split].to_pandas()

    # Preprocess data to only include records where "input" is not empty
    split_data = split_data[split_data["input"].apply(lambda x: len(x) > 0)]

    split_data.to_json(save_path, orient="records", lines=False)

