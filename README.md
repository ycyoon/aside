# ASIDE: Architecturally Separated Instruction-Data Embeddings
### This repo is under development, to be finished by June 6


[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2503.10566)
[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<p align="center">
  <img src="aside_teaser.jpg" width="70%" alt="ASIDE Overview">
</p>


**ASIDE** applies orthogonal rotations to data token embeddings while keeping instruction embeddings unchanged, creating clear separation without additional parameters or special safety training.

## ⚡ Quick Start

### Installation
```bash
git clone https://github.com/egozverev/aside.git
cd aside

# Create environment (Python 3.9-3.12)
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

### Training ASIDE
```bash
# Quick start: Train Qwen 2.5 7B with ASIDE
deepspeed --num_gpus=8 fine-tune.py \
--model_family qwen2.5_7b \
--train_version SFTv70 \
--emb_type forward_rot \
--model_ix 0 \
--run_number ASIDE \
--train_type full \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--learning_rate 2e-5 \
--lr_scheduler_type cosine \
--warmup_ratio 0 \
--logging_steps 10 \
--evaluation_strategy epoch \
--save_strategy epoch \
--eval_steps 1 \
--save_steps 1 \
--save_total_limit 1 \
--load_best_model_at_end True \
--prediction_loss_only True \
--bf16 True \
--embedding_init rot_isoclinic \
--rotation_alpha 1.57079633 \
--learned_rotation False \
--add_linear_shift False \
--rotation_direction right \
--gradual_rotation False

```

### Evaluation

1. **SEP Evaluation** - Instruction-data separation
   ```bash
   cd experiments/evaluations/SEP
   torchrun --nproc_per_node=1 --master_port=29706 get_model_outputs.py forward_rot qwen2.5_7b 1 SFTv70 forward_rot 15
   ```

2. **AlpacaEval** - General utility
   ```bash
   pip install -U alpaca_eval
   cd experiments/evaluations/AlpacaEval

   torchrun --nproc_per_node=1 --master_port=29712 get_alpaca_outputs.py --data-path data/tatsu-lab/alpaca_farm/eval.json --use-input True--model ../models/qwen2.5_7b/forward_rot/train_checkpoints/SFTv70/from_base_run_15/last/ --embedding-type forward_rot --batch-size 32

   IS_ALPACA_EVAL_2=False alpaca_eval --model_outputs ./data/tatsu-lab/alpaca_farm/qwen2.5_7b_forward_rot_train_checkpoints_SFTv70_from_base_run_15_last__l-1_s42.json
   ```

3. **Structured Queries** 
   ```bash
   cd experiments/evaluations/struq
   torchrun --nproc_per_node=1 --master_port=29718 test_on_struq.py --domain all --attack all --model ../models/qwen2.5_7b/forward_rot/train_checkpoints/SFTv70/from_base_run_15/last/ --embedding_type forward_rot --batch_size 32
   ```

4. **BIPIA**
TBD
5. **Other safety benchmarks**
TBD


Read `experiments/evaluations/README.md` for more details. 

## 🔧 Advanced Usage

### Generate Training and evaluation Scripts
```bash
# Create comprehensive hyperparameter sweep scripts
jupyter notebook scripts/generate_train_eval_scripts.ipynb
```
This notebook contains code that generates SLURM scripts for:
- Embedding types: `forward_rot` (ASIDE), `single_emb` (vanilla), `ise` (baseline)
- All supported models
- Different hyperparameters

Additionally, this notebook contain code that generates SLURM scripts for evals. 

### Analysis & Visualization
```bash
# Analyze results
jupyter notebook analysis/analyze_results.ipynb

# Generate paper figures
jupyter notebook analysis/visualization.ipynb
```

## 📁 Repository Structure
```
aside/
├── experiments/
│   ├── configs/           # Model configurations
│   ├── evaluations/       # Benchmarks: SEP, AlpacaEval, BIPIA, etc.
│   ├── interp/           # Interpretability analysis
│   └── fine-tune.py      # Main training script
├── analysis/
│   ├── analyze_results.py # Result processing
│   └── visualization.ipynb # Figure generation
└── scripts/
    └── generate_train_eval_scripts.ipynb
```


## 🛠️ Troubleshooting

- **OOM Errors**: Reduce `per_device_train_batch_size` in config
- **CUDA Issues**: Requires CUDA 12.4

## 📖 Citation

```bibtex

@inproceedings{zverev2025aside,
  author       = {Zverev, Egor and Kortukov, Evgenii and Panfilov, Alexander and Volkova, Alexandra and Tabesh, Soroush and Lapuschkin, Sebastian and Samek, Wojciech and Lampert, Christoph H.},
  title        = {ASIDE: Architectural Separation of Instructions and Data in Language Models},
  booktitle    = {ICLR 2025 Building Trust in LLMs and LLM Applications Workshop},
  year         = {2025},
}
```

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 📧 Contact

For questions or issues, please open a [GitHub issue](https://github.com/egozverev/aside/issues) or contact the authors through the paper correspondence.