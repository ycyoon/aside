# Reproducing the results in the Analysis section of the paper

## 6.1 Linear separability of representations

First we extract the activations of instruction and data  for all four models of interest:
```bash
python 6_extract_activations.py --model Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --embedding_type forward_rot --base_model meta-llama/Llama-3.1-8B
python 6_extract_activations.py --model Embeddings-Collab/llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --embedding_type ise --base_model meta-llama/Llama-3.1-8B
python 6_extract_activations.py --model Embeddings-Collab/llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --embedding_type single_emb --base_model meta-llama/Llama-3.1-8B
python 6_extract_activations.py --model meta-llama/Llama-3.1-8B --embedding_type base --base_model meta-llama/Llama-3.1-8B
```

Then we train a linear probing classifier for every layer:
```bash
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix/ --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix/ --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix/ --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Llama-3.1-8B/ --layer all
```

Finally, we plot the accuracies at each layer. It is saved to `token_classification.pdf` and `token_classification.png`
This step also prints all accuracies as a dataframe to the command line.
```bash
python 6_2_plot_classification.py --base Llama-3.1-8B --single llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --ise llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --double llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix 
```

## 6.2 Instruction feature activation

Here, we first need to extract the instruction feature from the alpaca dataset for each of our models.

```bash
python 2_1_probing.py --model_name Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix  --embedding_type forward_rot --base_model meta-llama/Llama-3.1-8B --template_type double --layer 15
python 2_1_probing.py --model_name Embeddings-Collab/llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --embedding_type ise --base_model meta-llama/Llama-3.1-8B  --template_type ise --layer 15
python 2_1_probing.py --model_name Embeddings-Collab/llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --embedding_type single_emb --base_model meta-llama/Llama-3.1-8B --template_type single --layer 15
python 2_1_probing.py  --model_name meta-llama/Llama-3.1-8B --embedding_type single_emb --base_model meta-llama/Llama-3.1-8B  --template_type base --layer 15
```

To generate latex code and a heatmap for the qualitative example:

ASIDE
```bash
python 2_3_qualitative_ex.py --model Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --embedding_type forward_rot --base_model meta-llama/Llama-3.1-8B  --layer 15
```

ISE
```bash
python 2_3_qualitative_ex.py --model Embeddings-Collab/llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --embedding_type ise --base_model meta-llama/Llama-3.1-8B  --layer 15
```

SFT
```bash
python 2_3_qualitative_ex.py --model Embeddings-Collab/llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --embedding_type single_emb --base_model meta-llama/Llama-3.1-8B  --layer 15
```

```bash
python 2_3_qualitative_ex.py --model meta-llama/Llama-3.1-8B --embedding_type base --base_model meta-llama/Llama-3.1-8B  --layer 15
```

To gather data for the histograms of the instruction feature activations:
```bash
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --embedding_type forward_rot --base_model meta-llama/Llama-3.1-8B  --layer 15
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --embedding_type ise --base_model meta-llama/Llama-3.1-8B  --layer 15
python 2_4_gather_feature_act_data.py   --model Embeddings-Collab/llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --embedding_type single_emb --base_model meta-llama/Llama-3.1-8B  --layer 15
python 2_4_gather_feature_act_data.py  --model meta-llama/Llama-3.1-8B --embedding_type base --base_model meta-llama/Llama-3.1-8B  --layer 15 
```

Finally, we plot the beautiful histograms and save them to file 6_2_combined_SEP_first1000.pdf
```bash
python 2_5_plot_histogram.py --base Llama-3.1-8B --single llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --ise llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --double llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --dataset full  --layer 15
```

For a short version with only Vanilla and Aside run the following command:
```bash
python 2_5_plot_histogram.py --base Llama-3.1-8B --single llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --ise llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --double llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --dataset full  --layer 15 --short-version
```

To do the same for injected and not injected datasets we do
```bash
python 2_5_plot_histogram.py --base Llama-3.1-8B --single llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --ise llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --double llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --dataset injected  --layer 15
python 2_5_plot_histogram.py --base Llama-3.1-8B --single llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --ise llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --double llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --dataset not_injected  --layer 15
```

## 6.3 Embedding interventions

Run the intervention experiment for the ASIDE model.
```bash
python 8_intervention.py --model Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --embedding_type forward_rot --base_model meta-llama/Llama-3.1-8B 

```

Paste the numbers you get from the previous step.
```bash
python 8_2_intervention_plot.py --clean 0.145 --intervention 0.278
```

## 6.4 Downstream effect of rotation

Gather the activations:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix --ise Embeddings-Collab/llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix  --double Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --base meta-llama/Llama-3.1-8B
```


Plot the results and save to file 6_4_rotation_downstream.pdf:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix  --ise Embeddings-Collab/llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix --double Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix --base meta-llama/Llama-3.1-8B --plot
```











# Llama 2-7B


## 6.1 Linear separability of representations

First we extract the activations of instruction and data  for all four models of interest:
```bash
python 6_extract_activations.py --model Embeddings-Collab/llama_2_7b_forward_rot_emb_SFTv110_from_base_run_6 --embedding_type forward_rot --base_model meta-llama/Llama-2-7b-hf
python 6_extract_activations.py --model Embeddings-Collab/llama_2_7b_ise_emb_SFTv110_from_base_run_35 --embedding_type ise --base_model meta-llama/Llama-2-7b-hf
python 6_extract_activations.py --model Embeddings-Collab/llama_2_7b_single_emb_emb_SFTv110_from_base_run_19 --embedding_type single_emb --base_model meta-llama/Llama-2-7b-hf
python 6_extract_activations.py --model meta-llama/Llama-2-7b-hf --embedding_type base --base_model meta-llama/Llama-2-7b-hf
```

Then we train a linear probing classifier for every layer:
```bash
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/llama_2_7b_forward_rot_emb_SFTv110_from_base_run_6 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/llama_2_7b_ise_emb_SFTv110_from_base_run_35 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/llama_2_7b_single_emb_emb_SFTv110_from_base_run_19 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Llama-2-7b-hf --layer all
```

Finally, we plot the accuracies at each layer. It is saved to `token_classification.pdf` and `token_classification.png`
This step also prints all accuracies as a dataframe to the command line.
```bash
python 6_2_plot_classification.py --base Llama-2-7b-hf --single llama_2_7b_single_emb_emb_SFTv110_from_base_run_19 --ise llama_2_7b_ise_emb_SFTv110_from_base_run_35 --double llama_2_7b_forward_rot_emb_SFTv110_from_base_run_6
```

## 6.2 Instruction feature activation

Here, we first need to extract the instruction feature from the alpaca dataset for each of our models.

```bash
python 2_1_probing.py --model_name Embeddings-Collab/llama_2_7b_forward_rot_emb_SFTv110_from_base_run_6  --embedding_type forward_rot --base_model meta-llama/Llama-2-7b-hf --template_type double --layer 15
python 2_1_probing.py --model_name Embeddings-Collab/llama_2_7b_ise_emb_SFTv110_from_base_run_35 --embedding_type ise --base_model meta-llama/Llama-2-7b-hf  --template_type ise --layer 15
python 2_1_probing.py --model_name Embeddings-Collab/llama_2_7b_single_emb_emb_SFTv110_from_base_run_19 --embedding_type single_emb --base_model meta-llama/Llama-2-7b-hf --template_type single --layer 15
python 2_1_probing.py --model_name meta-llama/Llama-2-7b-hf --embedding_type single_emb --base_model meta-llama/Llama-2-7b-hf  --template_type base --layer 15
```

To gather data for the histograms of the instruction feature activations:
```bash
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/llama_2_7b_forward_rot_emb_SFTv110_from_base_run_6 --embedding_type forward_rot --base_model meta-llama/Llama-2-7b-hf  --layer 15 --full_data_only
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/llama_2_7b_ise_emb_SFTv110_from_base_run_35 --embedding_type ise --base_model meta-llama/Llama-2-7b-hf  --layer 15 --full_data_only
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/llama_2_7b_single_emb_emb_SFTv110_from_base_run_19 --embedding_type single_emb --base_model meta-llama/Llama-2-7b-hf  --layer 15 --full_data_only
python 2_4_gather_feature_act_data.py --model meta-llama/Llama-2-7b-hf --embedding_type base --base_model meta-llama/Llama-2-7b-hf  --layer 15 --full_data_only
```

Finally, we plot the beautiful histograms with
```bash
python 2_5_plot_histogram.py --base Llama-2-7b-hf --single llama_2_7b_single_emb_emb_SFTv110_from_base_run_19 --ise llama_2_7b_ise_emb_SFTv110_from_base_run_35 --double llama_2_7b_forward_rot_emb_SFTv110_from_base_run_6 --dataset full  --layer 15
```

## 6.3 Embedding interventions

Run the intervention experiment for the ASIDE model.
```bash
python 8_intervention.py --model Embeddings-Collab/llama_2_7b_forward_rot_emb_SFTv110_from_base_run_6 --embedding_type forward_rot --base_model meta-llama/Llama-2-7b-hf

```

Paste the numbers you get from the previous step.
```bash
python 8_2_intervention_plot.py --clean 0.134 --intervention 0.216
```

## 6.4 Downstream effect of rotation

Gather the activations:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/llama_2_7b_single_emb_emb_SFTv110_from_base_run_19 --ise Embeddings-Collab/llama_2_7b_ise_emb_SFTv110_from_base_run_35 --double Embeddings-Collab/llama_2_7b_forward_rot_emb_SFTv110_from_base_run_6 --base meta-llama/Llama-2-7b-hf
```


Plot the results into the file 6_4_rotation_downstream.pdf:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/llama_2_7b_single_emb_emb_SFTv110_from_base_run_19 --ise Embeddings-Collab/llama_2_7b_ise_emb_SFTv110_from_base_run_35 --double Embeddings-Collab/llama_2_7b_forward_rot_emb_SFTv110_from_base_run_6 --base meta-llama/Llama-2-7b-hf --plot
```








# Llama 2-13B



## 6.1 Linear separability of representations

First we extract the activations of instruction and data  for all four models of interest:
```bash
python 6_extract_activations.py --model Embeddings-Collab/llama_2_13b_forward_rot_emb_SFTv110_from_base_run_15 --embedding_type forward_rot --base_model meta-llama/Llama-2-13b-hf
python 6_extract_activations.py --model Embeddings-Collab/llama_2_13b_ise_emb_SFTv110_from_base_run_36 --embedding_type ise --base_model meta-llama/Llama-2-13b-hf
python 6_extract_activations.py --model Embeddings-Collab/llama_2_13b_single_emb_emb_SFTv110_from_base_run_20 --embedding_type single_emb --base_model meta-llama/Llama-2-13b-hf
python 6_extract_activations.py --model meta-llama/Llama-2-13b-hf --embedding_type base --base_model meta-llama/Llama-2-13b-hf
```

Then we train a linear probing classifier for every layer:
```bash
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/llama_2_13b_forward_rot_emb_SFTv110_from_base_run_15 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/llama_2_13b_ise_emb_SFTv110_from_base_run_36 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/llama_2_13b_single_emb_emb_SFTv110_from_base_run_20 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Llama-2-13b-hf --layer all
```

Finally, we plot the accuracies at each layer. It is saved to `token_classification.pdf` and `token_classification.png`
This step also prints all accuracies as a dataframe to the command line.
```bash
python 6_2_plot_classification.py --base Llama-2-13b-hf --single llama_2_13b_single_emb_emb_SFTv110_from_base_run_20 --ise llama_2_13b_ise_emb_SFTv110_from_base_run_36 --double llama_2_13b_forward_rot_emb_SFTv110_from_base_run_15
```

## 6.2 Instruction feature activation

Here, we first need to extract the instruction feature from the alpaca dataset for each of our models.

```bash
python 2_1_probing.py --model_name Embeddings-Collab/llama_2_13b_forward_rot_emb_SFTv110_from_base_run_15 --embedding_type forward_rot --base_model meta-llama/Llama-2-13b-hf --template_type double --layer 18
python 2_1_probing.py --model_name Embeddings-Collab/llama_2_13b_ise_emb_SFTv110_from_base_run_36 --embedding_type ise --base_model meta-llama/Llama-2-13b-hf  --template_type ise --layer 18
python 2_1_probing.py --model_name Embeddings-Collab/llama_2_13b_single_emb_emb_SFTv110_from_base_run_20 --embedding_type single_emb --base_model meta-llama/Llama-2-13b-hf --template_type single --layer 18
python 2_1_probing.py --model_name meta-llama/Llama-2-13b-hf --embedding_type single_emb --base_model meta-llama/Llama-2-13b-hf  --template_type base --layer 18
```

To gather data for the histograms of the instruction feature activations:
```bash
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/llama_2_13b_forward_rot_emb_SFTv110_from_base_run_15 --embedding_type forward_rot --base_model meta-llama/Llama-2-13b-hf --layer 18 --full_data_only
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/llama_2_13b_ise_emb_SFTv110_from_base_run_36 --embedding_type ise --base_model meta-llama/Llama-2-13b-hf --layer 18 --full_data_only
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/llama_2_13b_single_emb_emb_SFTv110_from_base_run_20 --embedding_type single_emb --base_model meta-llama/Llama-2-13b-hf --layer 18 --full_data_only
python 2_4_gather_feature_act_data.py --model meta-llama/Llama-2-13b-hf --embedding_type base --base_model meta-llama/Llama-2-13b-hf --layer 18 --full_data_only
```

Finally, we plot the beautiful histograms with
```bash
python 2_5_plot_histogram.py --base Llama-2-13b-hf --single llama_2_13b_single_emb_emb_SFTv110_from_base_run_20 --ise llama_2_13b_ise_emb_SFTv110_from_base_run_36 --double llama_2_13b_forward_rot_emb_SFTv110_from_base_run_15 --dataset full --layer 18
```

## 6.3 Embedding interventions

Run the intervention experiment for the ASIDE model.
```bash
python 8_intervention.py --model Embeddings-Collab/llama_2_13b_forward_rot_emb_SFTv110_from_base_run_15 --embedding_type forward_rot --base_model meta-llama/Llama-2-13b-hf
```

Paste the numbers you get from the previous step.
```bash
python 8_2_intervention_plot.py --clean 0.179 --intervention 0.279
```


## 6.4 Downstream effect of rotation

Gather the activations:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/llama_2_13b_single_emb_emb_SFTv110_from_base_run_20 --ise Embeddings-Collab/llama_2_13b_ise_emb_SFTv110_from_base_run_36  --double Embeddings-Collab/llama_2_13b_forward_rot_emb_SFTv110_from_base_run_15 --base meta-llama/Llama-2-13b-hf
```


Plot the results:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/llama_2_13b_single_emb_emb_SFTv110_from_base_run_20 --ise Embeddings-Collab/llama_2_13b_ise_emb_SFTv110_from_base_run_36 --double Embeddings-Collab/llama_2_13b_forward_rot_emb_SFTv110_from_base_run_15 --base meta-llama/Llama-2-13b-hf --plot
```








# Qwen 2.5-7B

## 6.1 Linear separability of representations

First we extract the activations of instruction and data  for all four models of interest:
```bash
python 6_extract_activations.py --model Embeddings-Collab/Qwen2.5-7B_forward_rot_emb_SFTv70_from_inst_run_14 --embedding_type forward_rot --base_model Qwen/Qwen2.5-7B
python 6_extract_activations.py --model Embeddings-Collab/Qwen2.5-7B_ise_emb_SFTv70_from_inst_run_34 --embedding_type ise --base_model Qwen/Qwen2.5-7B
python 6_extract_activations.py --model Embeddings-Collab/Qwen2.5-7B_single_emb_emb_SFTv70_from_inst_run_18 --embedding_type single_emb --base_model Qwen/Qwen2.5-7B
python 6_extract_activations.py --model Qwen/Qwen2.5-7B --embedding_type base --base_model Qwen/Qwen2.5-7B
```

Then we train a linear probing classifier for every layer:
```bash
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Qwen2.5-7B_forward_rot_emb_SFTv70_from_inst_run_14 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Qwen2.5-7B_ise_emb_SFTv70_from_inst_run_34 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Qwen2.5-7B_single_emb_emb_SFTv70_from_inst_run_18 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Qwen2.5-7B --layer all
```

Finally, we plot the accuracies at each layer. It is saved to `token_classification.pdf` and `token_classification.png`
This step also prints all accuracies as a dataframe to the command line.
```bash
python 6_2_plot_classification.py --base Qwen2.5-7B --single Qwen2.5-7B_single_emb_emb_SFTv70_from_inst_run_18 --ise Qwen2.5-7B_ise_emb_SFTv70_from_inst_run_34 --double Qwen2.5-7B_forward_rot_emb_SFTv70_from_inst_run_14
```

## 6.2 Instruction feature activation
Here, we first need to extract the instruction feature from the alpaca dataset for each of our models.


```bash
python 2_1_probing.py --model_name Embeddings-Collab/Qwen2.5-7B_forward_rot_emb_SFTv70_from_inst_run_14 --embedding_type forward_rot --base_model Qwen/Qwen2.5-7B --template_type double --layer 14
python 2_1_probing.py --model_name Embeddings-Collab/Qwen2.5-7B_ise_emb_SFTv70_from_inst_run_34 --embedding_type ise --base_model Qwen/Qwen2.5-7B --template_type ise --layer 14
python 2_1_probing.py --model_name Embeddings-Collab/Qwen2.5-7B_single_emb_emb_SFTv70_from_inst_run_18 --embedding_type single_emb --base_model Qwen/Qwen2.5-7B --template_type single --layer 14
python 2_1_probing.py --model_name Qwen/Qwen2.5-7B --embedding_type single_emb --base_model Qwen/Qwen2.5-7B --template_type base --layer 14
```


To gather data for the histograms of the instruction feature activations:
```bash
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/Qwen2.5-7B_forward_rot_emb_SFTv70_from_inst_run_14 --embedding_type forward_rot --base_model Qwen/Qwen2.5-7B --layer 14 --full_data_only
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/Qwen2.5-7B_ise_emb_SFTv70_from_inst_run_34 --embedding_type ise --base_model Qwen/Qwen2.5-7B --layer 14 --full_data_only
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/Qwen2.5-7B_single_emb_emb_SFTv70_from_inst_run_18 --embedding_type single_emb --base_model Qwen/Qwen2.5-7B --layer 14 --full_data_only
python 2_4_gather_feature_act_data.py --model Qwen/Qwen2.5-7B --embedding_type base --base_model Qwen/Qwen2.5-7B --layer 14 --full_data_only
```

Finally, we plot the beautiful histograms and save them to 6_2_combined_SEP_first100.pdf with
```bash
python 2_5_plot_histogram.py --base Qwen2.5-7B --single Qwen2.5-7B_single_emb_emb_SFTv70_from_inst_run_18 --ise Qwen2.5-7B_ise_emb_SFTv70_from_inst_run_34 --double Qwen2.5-7B_forward_rot_emb_SFTv70_from_inst_run_14 --dataset full --layer 14
```

## 6.3 Embedding interventions
Run the intervention experiment for the ASIDE model.
```bash
python 8_intervention.py --model Embeddings-Collab/Qwen2.5-7B_forward_rot_emb_SFTv70_from_inst_run_14 --embedding_type forward_rot --base_model Qwen/Qwen2.5-7B
```
Paste the numbers you get from the previous step to plot to asr_comparison.pdf.
```bash
python 8_2_intervention_plot.py --clean 0.31  --intervention 0.466
```

## 6.4 Downstream effect of rotation

Gather the activations:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/Qwen2.5-7B_single_emb_emb_SFTv70_from_inst_run_18 --ise Embeddings-Collab/Qwen2.5-7B_ise_emb_SFTv70_from_inst_run_34 --double Embeddings-Collab/Qwen2.5-7B_forward_rot_emb_SFTv70_from_inst_run_14 --base Qwen/Qwen2.5-7B
```

Plot the results and save to file 6_4_rotation_downstream.pdf:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/Qwen2.5-7B_single_emb_emb_SFTv70_from_inst_run_18 --ise Embeddings-Collab/Qwen2.5-7B_ise_emb_SFTv70_from_inst_run_34 --double Embeddings-Collab/Qwen2.5-7B_forward_rot_emb_SFTv70_from_inst_run_14 --base Qwen/Qwen2.5-7B --plot
```








# Mistral 7B

## 6.1 Linear separability of representations

First we extract the activations of instruction and data  for all four models of interest:
```bash
python 6_extract_activations.py --model Embeddings-Collab/Mistral-7B-v0.3_forward_rot_emb_SFTv70_from_inst_run_10_new_pad_token --embedding_type forward_rot --base_model mistralai/Mistral-7B-v0.3
python 6_extract_activations.py --model Embeddings-Collab/Mistral-7B-v0.3_ise_emb_SFTv70_from_base_run_42 --embedding_type ise --base_model mistralai/Mistral-7B-v0.3
python 6_extract_activations.py --model Embeddings-Collab/Mistral-7B-v0.3_single_emb_emb_SFTv70_from_inst_run_16_new_pad_token --embedding_type single_emb --base_model mistralai/Mistral-7B-v0.3
python 6_extract_activations.py --model mistralai/Mistral-7B-v0.3 --embedding_type base --base_model mistralai/Mistral-7B-v0.3
```

Then we train a linear probing classifier for every layer:
```bash
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Mistral-7B-v0.3_forward_rot_emb_SFTv70_from_inst_run_10_new_pad_token --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Mistral-7B-v0.3_ise_emb_SFTv70_from_base_run_42 --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Mistral-7B-v0.3_single_emb_emb_SFTv70_from_inst_run_16_new_pad_token --layer all
python 6_1_id_classifier.py hidden_states_dp/alpaca_adv50percent/Mistral-7B-v0.3 --layer all
```

Finally, we plot the accuracies at each layer. It is saved to `token_classification.pdf` and `token_classification.png`
This step also prints all accuracies as a dataframe to the command line.
```bash
python 6_2_plot_classification.py --base Mistral-7B-v0.3 --single Mistral-7B-v0.3_single_emb_emb_SFTv70_from_inst_run_16_new_pad_token --ise Mistral-7B-v0.3_ise_emb_SFTv70_from_base_run_42 --double Mistral-7B-v0.3_forward_rot_emb_SFTv70_from_inst_run_10_new_pad_token
```


## 6.2 Instruction feature activation

Here, we first need to extract the instruction feature from the alpaca dataset for each of our models.

To choose layer, we run the probing on all layers for the base model
```bash
python 2_1_probing --model_name mistralai/Mistral-7B-v0.3 --embedding_type base --base_model mistralai/Mistral-7B-v0.3 --template_type base 
```

```bash
python 2_1_probing.py --model_name Embeddings-Collab/Mistral-7B-v0.3_forward_rot_emb_SFTv70_from_inst_run_10_new_pad_token --embedding_type forward_rot --base_model mistralai/Mistral-7B-v0.3 --template_type double --layer 15
python 2_1_probing.py --model_name Embeddings-Collab/Mistral-7B-v0.3_ise_emb_SFTv70_from_base_run_42 --embedding_type ise --base_model mistralai/Mistral-7B-v0.3 --template_type ise --layer 15
python 2_1_probing.py --model_name Embeddings-Collab/Mistral-7B-v0.3_single_emb_emb_SFTv70_from_inst_run_16_new_pad_token --embedding_type single_emb --base_model mistralai/Mistral-7B-v0.3 --template_type single --layer 15
python 2_1_probing.py --model_name mistralai/Mistral-7B-v0.3 --embedding_type single_emb --base_model mistralai/Mistral-7B-v0.3 --template_type base --layer 15
```

To gather data for the histograms of the instruction feature activations:
```bash
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/Mistral-7B-v0.3_forward_rot_emb_SFTv70_from_inst_run_10_new_pad_token --embedding_type forward_rot --base_model mistralai/Mistral-7B-v0.3 --layer 15 --full_data_only
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/Mistral-7B-v0.3_ise_emb_SFTv70_from_base_run_42 --embedding_type ise --base_model mistralai/Mistral-7B-v0.3 --layer 15 --full_data_only
python 2_4_gather_feature_act_data.py --model Embeddings-Collab/Mistral-7B-v0.3_single_emb_emb_SFTv70_from_inst_run_16_new_pad_token --embedding_type single_emb --base_model mistralai/Mistral-7B-v0.3 --layer 15 --full_data_only
python 2_4_gather_feature_act_data.py --model mistralai/Mistral-7B-v0.3 --embedding_type base --base_model mistralai/Mistral-7B-v0.3 --layer 15 --full_data_only
```

Finally, we plot the beautiful histograms and save them to file 6_2_combined_SEP_first100.pdf with
```bash
python 2_5_plot_histogram.py --base Mistral-7B-v0.3 --single Mistral-7B-v0.3_single_emb_emb_SFTv70_from_inst_run_16_new_pad_token --ise Mistral-7B-v0.3_ise_emb_SFTv70_from_base_run_42 --double Mistral-7B-v0.3_forward_rot_emb_SFTv70_from_inst_run_10_new_pad_token --dataset full --layer 15
```

## 6.3 Embedding interventions
Run the intervention experiment for the ASIDE model.
```bash
python 8_intervention.py --model Embeddings-Collab/Mistral-7B-v0.3_forward_rot_emb_SFTv70_from_inst_run_10_new_pad_token --embedding_type forward_rot --base_model mistralai/Mistral-7B-v0.3
```
Paste the numbers you get from the previous step.
```bash
python 8_2_intervention_plot.py --clean 0.048 --intervention 0.174
```


## 6.4 Downstream effect of rotation
Gather the activations:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/Mistral-7B-v0.3_single_emb_emb_SFTv70_from_inst_run_16_new_pad_token --ise Embeddings-Collab/Mistral-7B-v0.3_ise_emb_SFTv70_from_base_run_42 --double Embeddings-Collab/Mistral-7B-v0.3_forward_rot_emb_SFTv70_from_inst_run_10_new_pad_token --base mistralai/Mistral-7B-v0.3
```
Plot the results and save to file 6_4_rotation_downstream.pdf:
```bash
python 7_1_downstream_effect_of_rot.py --single Embeddings-Collab/Mistral-7B-v0.3_single_emb_emb_SFTv70_from_inst_run_16_new_pad_token --ise Embeddings-Collab/Mistral-7B-v0.3_ise_emb_SFTv70_from_base_run_42 --double Embeddings-Collab/Mistral-7B-v0.3_forward_rot_emb_SFTv70_from_inst_run_10_new_pad_token --base mistralai/Mistral-7B-v0.3 --plot
```







# Further models

To implement the analysis for other models, one would need to first implement this function
https://github.com/egozverev/side/blob/36da21b2708409a33d0315c9c4645a7662995852/model_api.py#L204
It computes values that depend on the template.
These values are used in this function https://github.com/egozverev/side/blob/36da21b2708409a33d0315c9c4645a7662995852/model_api.py#L366
to determine the tokens corresponding to instruction, data and probe.
To determine them one can maybe load a given model, run this function and play around with the values, looking at the function outputs, making sure that the 
return values `instruction_tokens_str`, `data_tokens_str` and `probe_tokens_str` are correct.