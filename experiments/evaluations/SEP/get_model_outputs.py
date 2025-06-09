"""
ASIDE Model Inference Script

This script generates model outputs for evaluation on the SEP (Should It Be Executed or Processed) 
dataset, which is used to compute instruction-data separation scores in the ASIDE paper.

The script supports multiple embedding types:
- forward_rot: ASIDE method (main contribution with π/2 rotation)  
- single_emb: Vanilla baseline method (standard embeddings)
- double_emb: Legacy double embedding approach
- ise: ISE baseline method

Usage:
    python get_model_outputs.py <embedding_type> <model_family> <model_ix> <train_v> <model> <run_n>
    
Example:
    srun --export=ALL torchrun --nproc_per_node=1 --master_port=29701 get_model_outputs.py ise llama_2_7b 1 SFTv110 ise 35
    srun --export=ALL torchrun --nproc_per_node=1 --master_port=29703 get_model_outputs.py forward_rot llama_2_7b 1 SFTv110 forward_rot 6
    srun --export=ALL torchrun --nproc_per_node=1 --master_port=29706 get_model_outputs.py single_emb llama_2_7b 1 SFTv110 pretrained_vanilla 19
The script generates JSON files with model predictions that are then used by the SEP 
evaluation pipeline to compute separation scores (Section 4.2 in the paper).

References:
    ASIDE: Architectural Separation of Instructions and Data in Language Models
    Section 4.2: Evaluation procedure
"""
import os.path
import sys

if "../.." not in sys.path:
    sys.path.append("../..")

from model_api import *
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def main(checkpoint_path: str, embedding_type: str, output_file_path:str, tokenizer_path:str,  chat_template_path:str, instruct_model_path = None, data_model_path = None,
         prompt_ix: int = 0, prompt_ix_end: Optional[int] = 1, 
         start_ix: Optional[int] = None, end_ix: Optional[int] = None) -> None:
    """
    Runs model inference on the SEP dataset to generate outputs for evaluation.
    
    This function loads a trained model (ASIDE, vanilla, or baseline) and generates
    predictions on the SEP dataset. The outputs are used to compute instruction-data
    separation scores as described in Section 4.2 of the ASIDE paper.
    
    Args:
        checkpoint_path (str): Path to the trained model checkpoint
        embedding_type (str): Type of embedding method:
                             - "forward_rot": ASIDE method (main contribution)
                             - "single_emb": Vanilla baseline 
                             - "double_emb": Legacy double embedding
                             - "ise": ISE baseline
        output_file_path (str): Path to save the inference results (JSON format)
        tokenizer_path (str): Path to the tokenizer (usually instruction-tuned version)
        chat_template_path (str): Path to chat template file
        instruct_model_path (str, optional): Path to instruction model (for double_emb)
        data_model_path (str, optional): Path to data model (for double_emb)
        prompt_ix (int): Starting prompt template index (default: 0)
        prompt_ix_end (int, optional): Ending prompt template index (default: 1)
        start_ix (int, optional): Starting dataset index for evaluation (default: 7000)
        end_ix (int, optional): Ending dataset index for evaluation (default: 8000)
        
    Note:
        - Uses SEP dataset from "../../data/SEP_dataset.json"
        - Prompt templates from "../../data/prompt_templates.json"
        - Batch size is set to 32 for efficiency
        
    The generated outputs follow the SEP evaluation protocol where models are prompted
    with (instruction, data) pairs containing probe strings in either instruction or
    data segments. Success is measured by whether models execute probes only when
    they appear in instruction segments.

    References: 
        Can LLMs Separate Instructions From Data? And What Do We Even Mean By That?
    """
    if embedding_type == "double_emb":
        batch_size=32
    else:
        batch_size=32
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    if prompt_ix_end is None:
        prompt_ix_end = prompt_ix + 1
    if prompt_ix == prompt_ix_end:
        raise Exception("Prompt index interval is empty")

    for p_ix in range(prompt_ix, prompt_ix_end):
        dataset, prompt_template = load_data("../../data/SEP_dataset.json", "../../data/prompt_templates.json", p_ix)
        if start_ix is None:
            start_ix = 0
        if end_ix is None:
            end_ix = len(dataset)
        dataset = dataset[start_ix: end_ix]
        template_info = {"template_prompt_ix": p_ix, "template_prompt": prompt_template}
        model_dtype = torch.bfloat16
        handler = CustomModelHandler(checkpoint_path, instruct_model_path, data_model_path, tokenizer_path, chat_template_path,
                                     prompt_ix, embedding_type=embedding_type, load_from_checkpoint=True,
                                     model_dtype=model_dtype)
        print(f"Starting inference for model {checkpoint_path} on prompt index {p_ix}. \
              Dataset slice is dataset[{start_ix}:{end_ix}]")
        
        inference(dataset, output_file_path, template_info, handler, batch_size=batch_size, mp_size=1)

        print(f"Inference complete. Results saved to {output_file_path}")



"""
    Path configurations for double embedding models (legacy approach).
    
    Structure:
    - instruct_model_path: Base model for instruction embeddings
    - data_model_path: Base model for data embeddings (usually same as instruct)
    - tokenizer_path: Instruction-tuned tokenizer for proper formatting
"""
DOUBLE_EMB_PATHS = {
    "llama_3.1_8b": {
        "instruct_model_path": "meta-llama/Llama-3.1-8B",
        "data_model_path":     "meta-llama/Llama-3.1-8B",
        "tokenizer_path":      "meta-llama/Llama-3.1-8B-Instruct",
    },
    "llama_2_7b": {
        "instruct_model_path": "meta-llama/Llama-2-7b-hf",
        "data_model_path":     "meta-llama/Llama-2-7b-hf",
        "tokenizer_path":      "meta-llama/Llama-2-7b-chat-hf",
    },
    "llama_2_13b": {
        "instruct_model_path": "meta-llama/Llama-2-13b-hf",
        "data_model_path":     "meta-llama/Llama-2-13b-hf",
        "tokenizer_path":      "meta-llama/Llama-2-13b-chat-hf",
    },
    "Qwen2.5-7B": {
        "instruct_model_path": "Qwen/Qwen2.5-7B",
        "data_model_path": "Qwen/Qwen2.5-7B",
        "tokenizer_path": "Qwen/Qwen2.5-7B-Instruct",
    },
    "Mistral-7B-v0.3": {
        "instruct_model_path": "mistralai/Mistral-7B-v0.3",
        "data_model_path": "mistralai/Mistral-7B-v0.3",
        "tokenizer_path": "mistralai/Mistral-7B-Instruct-v0.3",
    }
}

"""
    Path configurations for single embedding models (vanilla baselines and original models).
    
    Structure:
    - original: Base pretrained model (vanilla baseline)
    - original_inst: Instruction-tuned model (vanilla baseline)
    - These are used for baseline comparisons against ASIDE methods
"""
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

def get_double_emb_paths(model_family):
    """
    Retrieves model paths for double embedding configurations.
    
    Double embedding uses separate embedding matrices for instruction and data tokens.
    This is a legacy approach superseded by ASIDE's rotation method.
    
    Args:
        model_family (str): Model family identifier (e.g., "llama_3.1_8b")
        
    Returns:
        tuple: (instruct_model_path, data_model_path, tokenizer_path)
        
    Raises:
        ValueError: If model_family is not supported for double_emb
        
    Example:
        >>> paths = get_double_emb_paths("llama_3.1_8b")
        >>> print(paths)
        ('meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-8B-Instruct')
    """
    if model_family not in DOUBLE_EMB_PATHS:
        raise ValueError(f"Unknown model family for double_emb: {model_family}")
    paths = DOUBLE_EMB_PATHS[model_family]
    return (
        paths["instruct_model_path"],
        paths["data_model_path"],
        paths["tokenizer_path"],
    )


def get_single_emb_paths(model_family, train_v, model):
    """
    Retrieves model paths for single embedding configurations (Vanilla, ISE, ASIDE)
    
    Args:
        model_family (str): Model family identifier (e.g., "llama_3.1_8b")
        train_v (str): Training version identifier 
        model (str): Specific model variant 
        
    Returns:
        tuple: (checkpoint_path, output_suggested_path) or (None, None) if not found
        
    Note:
        - "original": Base pretrained model
        - "original_inst": Instruction-tuned model  
        - These models use standard embeddings (no rotation or separation)
        
    Example:
        >>> checkpoint, output = get_single_emb_paths("llama_3.1_8b", "SFTv11", "original")
        >>> print(checkpoint)
        'meta-llama/Llama-3.1-8B'
    """
    if model_family not in SINGLE_EMB_PATHS:
        return None, None
    
    family_dict = SINGLE_EMB_PATHS[model_family]
    print(family_dict.items(), model)
    checkpoint_path = family_dict.get(model, None)
    print(checkpoint_path)
    if checkpoint_path is None:
        return None, None
    
    # Construct an output file path by convention, but the caller
    # can still override if needed.
    output_file_path = f"./model_outputs/{model_family}/{train_v}/{model}_fullsep.json"
    return checkpoint_path, output_file_path


if __name__ == "__main__":
    """
    Command-line interface for running model inference on SEP dataset.
    
    Arguments:
        1. embedding_type: "forward_rot" (ASIDE), "single_emb" (vanilla), "double_emb" (legacy), or "ise" (ISE)
        2. model_family: Model family (e.g., "llama_3.1_8b")
        3. model_ix: Model index ("0" for from_inst, "1" for from_base) -- in all out experiments, model_ix == 1
        4. train_v: Training version (e.g., "SFTv110")
        5. model: Model variant name (e.g., forward_rot (ASIDE), ise (ISE), pretrained_vanilla (Vanilla))
        6. run_n: Run number for experiment tracking
        
    Examples:
        srun --export=ALL torchrun --nproc_per_node=1 --master_port=29701 get_model_outputs.py ise llama_2_7b 1 SFTv110 ise 35
        srun --export=ALL torchrun --nproc_per_node=1 --master_port=29703 get_model_outputs.py forward_rot llama_2_7b 1 SFTv110 forward_rot 6
        srun --export=ALL torchrun --nproc_per_node=1 --master_port=29706 get_model_outputs.py single_emb llama_2_7b 1 SFTv110 pretrained_vanilla 19
    """
    #os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    if len(sys.argv) != 7:
        raise ValueError("Expected 6 additional arguments after script name.")
    
    embedding_type, model_family, model_ix, train_v, model, run_n = sys.argv[1:] #.e.g., single_emb  llama_3.1_8b 1 SFTv110 pretrained_vanilla 14 
    model_type = "from_inst" if (model_ix == "0") else "from_base"
    instruct_model_path = None
    data_model_path = None
    tokenizer_path = None
    
    if embedding_type == "double_emb":
        checkpoint_path = (
            f"../../models/{model_family}/{model}/train_checkpoints/"
            f"{train_v}/{model_type}_run_{run_n}/last"
        )
        output_file_path = (
            f"../../model_outputs/{model_family}/{train_v}/"
            f"{model}_{model_type}_run{run_n}_fullsep.json"
        )
        
        # Retrieve the triple paths from the dictionary
        (instruct_model_path,
         data_model_path,
         tokenizer_path) = get_double_emb_paths(model_family)
    elif embedding_type in ("ise", "forward_rot"):
        checkpoint_path = (
            f"../../models/{model_family}/{model}/train_checkpoints/"
            f"{train_v}/{model_type}_run_{run_n}/last"
        )
        output_file_path = (
            f"../../model_outputs/{model_family}/{train_v}/"
            f"{model}_{model_type}_run{run_n}_fullsep.json"
        )
        
        # Retrieve the triple paths from the dictionary
        _, _, tokenizer_path = get_double_emb_paths(model_family)
        instruct_model_path, data_model_path = None, None
    elif embedding_type == "single_emb":
        single_cp_path, single_out_path = get_single_emb_paths(model_family, train_v, model)
        print("paths", single_cp_path, single_cp_path)

        if single_cp_path is not None:
            # If we found a direct path, use it
            checkpoint_path = single_cp_path
            # Insert the actual train_v into the output path
            output_file_path = single_out_path.format(train_v=train_v)
            
        else:
            # Otherwise, fallback to a standard checkpoint path
            checkpoint_path = (
                f"../../models/{model_family}/{model}/train_checkpoints/"
                f"{train_v}/{model_type}_run_{run_n}/last"
            )
            output_file_path = (
                f"../../model_outputs/{model_family}/{train_v}/"
                f"{model_type}_{model}_run{run_n}_fullsep.json"
            )
            
        
        # For single-emb, we do not have distinct instruct/data model paths,
        # so just point the tokenizer to the same path as the checkpoint.
        tokenizer_path = checkpoint_path

        print(f"Tokenizer {tokenizer_path}")
    else:
        raise NotImplementedError("Only implemented for single, double, forward_rot and ISE")

    chat_template_path = None
    main(checkpoint_path=checkpoint_path, embedding_type=embedding_type,output_file_path=output_file_path, tokenizer_path=tokenizer_path,
         chat_template_path=chat_template_path, instruct_model_path=instruct_model_path, data_model_path=data_model_path)


