# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT License.

# from collections import OrderedDict
# from pathlib import Path

# import yaml
# from accelerate.logging import get_logger

# # from .gpt import GPT35, GPT4, GPT35WOSystem, GPT4WOSystem
# from .llama import (
#     GPT4ALL,
#     Alpaca,
#     Baize,
#     Guanaco,
#     Koala,
#     Llama2,
#     StableVicuna,
#     Vicuna,
#     Wizard,
# )
# from .llm_worker import OASST, ChatGLM, FastChatT5, RwkvModel
# from .vllm_worker import MPT, Dolly, Mistral, StableLM

# logger = get_logger(__name__)

# LLM_NAME_TO_CLASS = OrderedDict(
#     [
#         ("gpt35", GPT35),
#         ("gpt4", GPT4),
#         ("gpt35_wosys", GPT35WOSystem),
#         ("gpt4_wosys", GPT4WOSystem),
#         ("alpaca", Alpaca),
#         ("vicuna", Vicuna),
#         ("baize", Baize),
#         ("stablelm", StableLM),
#         ("stablevicuna", StableVicuna),
#         ("dolly", Dolly),
#         ("rwkv", RwkvModel),
#         ("oasst", OASST),
#         ("chatglm", ChatGLM),
#         ("koala", Koala),
#         ("mpt", MPT),
#         ("t5", FastChatT5),
#         ("gpt4all", GPT4ALL),
#         ("wizard", Wizard),
#         ("guanaco", Guanaco),
#         ("llama2", Llama2),
#         ("mistral", Mistral),
#     ]
# )


# class AutoLLM:
#     @classmethod
#     def from_name(cls, name: str):
#         if name in LLM_NAME_TO_CLASS:
#             name = name
#         elif Path(name).exists():
#             with open(name, "r") as f:
#                 config = yaml.load(f, Loader=yaml.SafeLoader)
#             if "llm_name" not in config:
#                 raise ValueError("llm_name not in config.")
#             name = config["llm_name"]
#         else:
#             raise ValueError(
#                 f"Invalid name {name}. AutoLLM.from_name needs llm name or llm config as inputs."
#             )

#         logger.info(f"Load {name} from name.")

#         llm_cls = LLM_NAME_TO_CLASS[name]
#         return llm_cls
