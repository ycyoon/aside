
# AlpacaEval


### Get the data.

```
cd evals
python download_alpaca_data.py --dataset tatsu-lab/alpaca_eval
python download_alpaca_data.py --dataset tatsu-lab/alpaca_farm --subset alpaca_farm_evaluation
```

### Generate outputs on the alpaca eval set.
On single model (all goes into instruction).
```
python get_alpaca_outputs.py --data-path data/tatsu-lab/alpaca_eval/eval.json  --model Embeddings-Collab/llama_3.1_8b_single_emb_SFTv50_from_base_run_5e-6_bs8  --embedding-type single_emb --base-model meta-llama/Llama-3.1-8B --batch-size 32
```
I usually add `CUDA_VISIBLE_DEVICES=5` at the beginning to specify a gpu to use.

On the double model (note the `alpaca_farm` dataset and `--use-input` flag)
```
python get_alpaca_outputs.py --data-path data/tatsu-lab/alpaca_farm/eval.json --use-input True --model Embeddings-Collab/llama_3.1_8b_double_emb_SFTv50_from_base_run_5e-6_bs8  --embedding-type double_emb --base-model meta-llama/Llama-3.1-8B --batch-size 32
```

### Evaluate the outputs.
To get alpaca 1.0 the env variable is used.
The path to model outputs will be given by the previous command.
Note, this step costs money.
```
IS_ALPACA_EVAL_2=False alpaca_eval --model_outputs data/tatsu-lab/alpaca_farm/llama_3.1_8b_double_emb_SFTv50_from_inst_run_5e-6_bs8_l-1_s42.json
```


# IF-Eval
```
python if_eval.py --model meta-llama/Llama-3.1-8B  --embedding-type single_emb --base-model meta-llama/Llama-3.1-8B --batch-size 32
```

# StruQ
### Get the data.
```
cd struq
python download_struq_data.py
```

### Evaluate the model
```
python test_on_struq.py --domain all --attack all --model Embeddings-Collab/llama_3.1_8b_double_emb_SFTv50_from_base_run_5e-6_bs8  --embedding_type double_emb --base_model meta-llama/Llama-3.1-8B --batch_size 32
```


# Safety metrics
```
python test_safety.py --model_name meta-llama/Llama-3.1-8B-Instruct --embedding_type single_emb --base_model meta-llama/Llama-3.1-8B-Instruct
```
