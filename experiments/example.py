import torch
import deepspeed
import json
import os
from huggingface_hub import login

from model_api import CustomModelHandler  # Import your custom handler
from model_api import format_prompt  # Import your prompt formatting function

# Define your instruction and data
instruction_text = "Translate to German."
data_text = "Who is Albert Einstein?"

# Model configuration
hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
login(token=hf_token)
embedding_type = "forward_rot"  # or "single_emb", "ise"
base_model =  "Qwen/Qwen2.5-7B" #or "meta-llama/Llama-3.1-8B"  #others
model_path = "path_to_your_model"

# Initialize the model handler
handler = CustomModelHandler(
    model_path, 
    base_model, 
    base_model, 
    model_path, 
    None,
    0, 
    embedding_type=embedding_type, 
    load_from_checkpoint=True
)

# Initialize DeepSpeed inference engine
engine = deepspeed.init_inference(
    model=handler.model,
    mp_size=torch.cuda.device_count(),  # Number of GPUs
    dtype=torch.float16,
    replace_method='auto',
    replace_with_kernel_inject=False
)
handler.model = engine.module

# Load prompt templates
with open("./data/prompt_templates.json", "r") as f:
    templates = json.load(f)

template = templates[0]  
instruction_text = format_prompt(instruction_text, template, "system")
data_text = format_prompt(data_text, template, "user")

# Generate output
output, inp = handler.call_model_api_batch([instruction_text], [data_text])
print(output)