### OLD, NO REDUCE OPERATIONS USED
import os
import sys
import torch


# from accelerate import dispatch_model
# from accelerate.utils import convert_outputs_to_fp32

# import deepspeed
# import torch.distributed as dist

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the higher directory to sys.path
sys.path.insert(0, project_dir)

from model_api import *


def get_embedding_matrices(double_embedding_model):
    """Get the cloned embedding matrix from the custom model handler.

    Args:
        double_embedding_model: A model that, where embedding matrix size is (2 * actual vocab_size).
            First matrix is for instruction, second for data.
    Returns:
        (tuple): A tuple of two torch.Tensor objects, the first for instruction embeddings, the second for data embeddings.
    """

    declared_vocab_size = double_embedding_model.config.declared_vocab_size
    specials_vocab_size = double_embedding_model.config.specials_vocab_size

    # with FSDP.summon_full_params(double_embedding_model):
    #     inst_embd = double_embedding_model.model.embed_tokens.weight[:actual_vocab_size, :].cpu()
    #     data_embd = double_embedding_model.model.embed_tokens.weight[actual_vocab_size:, :].cpu()
    #
    # with deepspeed.zero.GatheredParameters(
    #     [double_embedding_model.model.embed_tokens.weight],
    #     modifier_rank=0,
    #     enabled=dist.get_world_size() > 1,  # or always True if you're distributed
    # ):
    #     if dist.get_rank() == 0:

    # On rank=0, it's fully gathered, so we can slice freely:
    # Cloning so we can delete the model from memory but keep the embeddings.
    inst_embd = double_embedding_model.model.embed_tokens.weight[:declared_vocab_size, :].float().detach().cpu().clone()
    data_embd = double_embedding_model.model.embed_tokens.weight[declared_vocab_size + specials_vocab_size:, :].float().detach().cpu().clone()
    # else:
    #     # On other ranks, double_embedding_model.model.embed_tokens.weight
    #     # will remain sharded or be set to None
    #     inst_embd = None
    #     data_embd = None
    # Outside the context, the sharding is re-applied.
    return inst_embd, data_embd



def get_embedding_metrics(first_embd, second_embd):
    """Calculate metrics on how instruction and data embeddings differ.
    
    Args:
        first_embd (torch.Tensor): The first embedding matrix.
        second_embd (torch.Tensor): The second embedding matrix

    Returns:
        (dict): A dictionary with similarity metrics.
    """

    assert first_embd.size() == second_embd.size(), "Embeddings must have the same size"


    # Cosine similarity
    print("Loaded embeddings")
    cos_sims = torch.nn.functional.cosine_similarity(first_embd, second_embd, dim=-1)
    cos_sim_dict = {
        "mean": cos_sims.mean().item(),
        "std": cos_sims.std().item(),
        "median": torch.median(cos_sims).item(),
        "min": cos_sims.min().item(),
        "max": cos_sims.max().item(),
        "percentile_5": torch.quantile(cos_sims, 0.05).item(),
        "percentile_10": torch.quantile(cos_sims, 0.10).item(),
        "percentile_90": torch.quantile(cos_sims, 0.90).item(),
        "percentile_95": torch.quantile(cos_sims, 0.95).item(),
    }
    print("Computed cosine")

    l2_dist = torch.nn.functional.pairwise_distance(first_embd, second_embd, p=2)
    l2_dist_dict = {
        "mean": l2_dist.mean().item(),
        "std": l2_dist.std().item(),
        "median": torch.median(l2_dist).item(),
        "min": l2_dist.min().item(),
        "max": l2_dist.max().item(),
        "percentile_5": torch.quantile(l2_dist, 0.05).item(),
        "percentile_10": torch.quantile(l2_dist, 0.10).item(),
        "percentile_90": torch.quantile(l2_dist, 0.90).item(),
        "percentile_95": torch.quantile(l2_dist, 0.95).item(),
    }
    print("Computed l2")

    l_inf_dist = torch.nn.functional.pairwise_distance(first_embd, second_embd, p=float('inf'))
    l_inf_dist_dict = {
        "mean": l_inf_dist.mean().item(),
        "std": l_inf_dist.std().item(),
        "median": torch.median(l_inf_dist).item(),
        "min": l_inf_dist.min().item(),
        "max": l_inf_dist.max().item(),
        "percentile_5": torch.quantile(l_inf_dist, 0.05).item(),
        "percentile_10": torch.quantile(l_inf_dist, 0.10).item(),
        "percentile_90": torch.quantile(l_inf_dist, 0.90).item(),
        "percentile_95": torch.quantile(l_inf_dist, 0.95).item(),
    }
    print("Computed l_inf")

    return {
        "cosine_similarity": cos_sim_dict,
        "l2_distance": l2_dist_dict,
        "l_inf_distance": l_inf_dist_dict,
    }

if __name__ == "__main__":
    embeddings_init = "copy"
    alpha = torch.pi / 2  #[[0.14, 0.315, 0.79, 1.57]]

    embedding_type = "double_emb"


    # checkpoint_path = "./models/llama_3.1_8b/dd_pure/train_checkpoints/SFTv40/from_base_run_5e-6_bs8/last"
    # short_model_name = checkpoint_path.split("/")[-2]

    # base_model_path = "meta-llama/Llama-3.1-8B"
    # save_dir = "interp/emb_metrics"
    # checkpoint_path = "Embeddings-Collab/llama_3.1_8b_double_emb_SFTv50_from_base_run_5e-6_bs8"
    checkpoint_path = "Embeddings-Collab/llama_3.1_8b_double_emb_SFTv50_from_base_run_5e-6_bs8_norotation"
    short_model_name = checkpoint_path.split("/")[-1]

    base_model_path = "meta-llama/Llama-3.1-8B"
    save_dir = "emb_metrics"
    
    save_path = os.path.join(save_dir,  f"{short_model_name}.json")

    os.makedirs(save_dir, exist_ok=True)
    print("Starting evaluation init embeddings")

    handler = CustomModelHandler(
        checkpoint_path, base_model_path, base_model_path, checkpoint_path, None, embedding_type=embedding_type,
        embeddings_init=embeddings_init,
        rotation_alpha=alpha,
        model_dtype=torch.bfloat16,
    )
    inst_embd_init, data_embd_init = get_embedding_matrices(handler.model)
    del handler
    metrics_init = get_embedding_metrics(inst_embd_init, data_embd_init)
    print("Evaluated init embeddings")
    print("Starting evaluation trained embeddings")

    handler = CustomModelHandler(
        checkpoint_path, base_model_path, base_model_path, checkpoint_path, None, embedding_type=embedding_type,
        load_from_checkpoint=True
    )
    inst_embd_trained, data_embd_trained = get_embedding_matrices(handler.model)
    del handler
    metrics_trained = get_embedding_metrics(inst_embd_trained, data_embd_trained)
    print("Evaluated trained embeddings")

    print("Starting evaluation before-after inst embeddings")
    metrics_before_after_inst = get_embedding_metrics(inst_embd_init, inst_embd_trained)
    print("Evaluated before-after inst embeddings")

    print("Starting evaluation before-after data embeddings")
    metrics_before_after_data = get_embedding_metrics(data_embd_init, data_embd_trained)
    print("Evaluated before-after data embeddings")

    print("Starting evaluation before-after full embeddings")
    full_embd_init = torch.cat([inst_embd_init, data_embd_init], dim=0)
    full_embd_trained = torch.cat([inst_embd_trained, data_embd_trained], dim=0)
    metrics_before_after_full = get_embedding_metrics(full_embd_init, full_embd_trained)
    print("Evaluated before-after full embeddings")

    with open(save_path, "w+") as f:
        json.dump(
            {
                "metrics_inst/data_init": metrics_init,
                "metrics_inst/data_trained": metrics_trained,
                "metrics_before_after_inst": metrics_before_after_inst,
                "metrics_before_after_data": metrics_before_after_data,
                "metrics_before_after_full": metrics_before_after_full,
            },
            f,
            indent=2,
        )


# ### OLD, NO REDUCE OPERATIONS USED
# import os
# import sys
# import torch


# # from accelerate import dispatch_model
# # from accelerate.utils import convert_outputs_to_fp32

# # import deepspeed
# # import torch.distributed as dist

# project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Add the higher directory to sys.path
# sys.path.insert(0, project_dir)

# from model_api import *


# def get_embedding_matrices(double_embedding_model):
#     """Get the cloned embedding matrix from the custom model handler.

#     Args:
#         double_embedding_model: A model that, where embedding matrix size is (2 * actual vocab_size).
#             First matrix is for instruction, second for data.
#     Returns:
#         (tuple): A tuple of two torch.Tensor objects, the first for instruction embeddings, the second for data embeddings.
#     """

#     declared_vocab_size = double_embedding_model.config.declared_vocab_size
#     specials_vocab_size = double_embedding_model.config.specials_vocab_size

#     # with FSDP.summon_full_params(double_embedding_model):
#     #     inst_embd = double_embedding_model.model.embed_tokens.weight[:actual_vocab_size, :].cpu()
#     #     data_embd = double_embedding_model.model.embed_tokens.weight[actual_vocab_size:, :].cpu()
#     #
#     # with deepspeed.zero.GatheredParameters(
#     #     [double_embedding_model.model.embed_tokens.weight],
#     #     modifier_rank=0,
#     #     enabled=dist.get_world_size() > 1,  # or always True if you're distributed
#     # ):
#     #     if dist.get_rank() == 0:

#     # On rank=0, it's fully gathered, so we can slice freely:
#     # Cloning so we can delete the model from memory but keep the embeddings.
#     inst_embd = double_embedding_model.model.embed_tokens.weight[:declared_vocab_size, :].float().detach().cpu().clone()
#     data_embd = double_embedding_model.model.embed_tokens.weight[declared_vocab_size + specials_vocab_size:, :].float().detach().cpu().clone()
#     # else:
#     #     # On other ranks, double_embedding_model.model.embed_tokens.weight
#     #     # will remain sharded or be set to None
#     #     inst_embd = None
#     #     data_embd = None
#     # Outside the context, the sharding is re-applied.
#     return inst_embd, data_embd



# def get_embedding_metrics(first_embd, second_embd):
#     """Calculate metrics on how instruction and data embeddings differ.
    
#     Args:
#         first_embd (torch.Tensor): The first embedding matrix.
#         second_embd (torch.Tensor): The second embedding matrix

#     Returns:
#         (dict): A dictionary with similarity metrics.
#     """

#     assert first_embd.size() == second_embd.size(), "Embeddings must have the same size"


#     # Cosine similarity
#     print("Loaded embeddings")
#     cos_sims = torch.nn.functional.cosine_similarity(first_embd, second_embd, dim=-1)
#     cos_sim_dict = {
#         "mean": cos_sims.mean().item(),
#         "std": cos_sims.std().item(),
#         "median": torch.median(cos_sims).item(),
#         "min": cos_sims.min().item(),
#         "max": cos_sims.max().item(),
#         "percentile_5": torch.quantile(cos_sims, 0.05).item(),
#         "percentile_10": torch.quantile(cos_sims, 0.10).item(),
#         "percentile_90": torch.quantile(cos_sims, 0.90).item(),
#         "percentile_95": torch.quantile(cos_sims, 0.95).item(),
#     }
#     print("Computed cosine")

#     l2_dist = torch.nn.functional.pairwise_distance(first_embd, second_embd, p=2)
#     l2_dist_dict = {
#         "mean": l2_dist.mean().item(),
#         "std": l2_dist.std().item(),
#         "median": torch.median(l2_dist).item(),
#         "min": l2_dist.min().item(),
#         "max": l2_dist.max().item(),
#         "percentile_5": torch.quantile(l2_dist, 0.05).item(),
#         "percentile_10": torch.quantile(l2_dist, 0.10).item(),
#         "percentile_90": torch.quantile(l2_dist, 0.90).item(),
#         "percentile_95": torch.quantile(l2_dist, 0.95).item(),
#     }
#     print("Computed l2")

#     l_inf_dist = torch.nn.functional.pairwise_distance(first_embd, second_embd, p=float('inf'))
#     l_inf_dist_dict = {
#         "mean": l_inf_dist.mean().item(),
#         "std": l_inf_dist.std().item(),
#         "median": torch.median(l_inf_dist).item(),
#         "min": l_inf_dist.min().item(),
#         "max": l_inf_dist.max().item(),
#         "percentile_5": torch.quantile(l_inf_dist, 0.05).item(),
#         "percentile_10": torch.quantile(l_inf_dist, 0.10).item(),
#         "percentile_90": torch.quantile(l_inf_dist, 0.90).item(),
#         "percentile_95": torch.quantile(l_inf_dist, 0.95).item(),
#     }
#     print("Computed l_inf")

#     return {
#         "cosine_similarity": cos_sim_dict,
#         "l2_distance": l2_dist_dict,
#         "l_inf_distance": l_inf_dist_dict,
#     }

# if __name__ == "__main__":
#     train_v = "SFTv17"
#     model = "dd"
#     run_n = 0
#     checkpoint_n = "checkpoint-350"#"last"
#     embeddings_init = "copy"
#     alpha = torch.pi / 2  #[[0.14, 0.315, 0.79, 1.57]]

#     embedding_type = "double_emb"
#     checkpoint_path = f"./models/llama_3.1_8b/{model}_pure/train_checkpoints/{train_v}/run_{run_n}/{checkpoint_n}"
#     #checkpoint_path = f"./models/llama_3.1_8b/{model}_pure/train_checkpoints/{train_v}/run_{run_n}/checkpoint-{checkpoint_n}"

#     #checkpoint_path = f"../models/Embeddings-Collab/llama_3.1_8b_double_emb_SFTv19_run_7"

#     instruct_model_path = "meta-llama/Llama-3.1-8B"
#     data_model_path = "meta-llama/Llama-3.1-8B"
#     save_dir = os.path.join("interp","emb_metrics", train_v)
#     save_path = os.path.join(save_dir,  f"{model}_{run_n}_{checkpoint_n}.json")
#     os.makedirs(save_dir, exist_ok=True)
#     print("Starting evaluation init embeddings")

#     handler = CustomModelHandler(
#         checkpoint_path, instruct_model_path, data_model_path, None, embedding_type=embedding_type,
#         embeddings_init=embeddings_init,
#         rotation_alpha=alpha
#     )
#     inst_embd_init, data_embd_init = get_embedding_matrices(handler.model)
#     del handler
#     metrics_init = get_embedding_metrics(inst_embd_init, data_embd_init)
#     print("Evaluated init embeddings")
#     print("Starting evaluation trained embeddings")

#     handler = CustomModelHandler(
#         checkpoint_path, instruct_model_path, data_model_path, None, embedding_type=embedding_type,
#         load_from_checkpoint=True
#     )
#     inst_embd_trained, data_embd_trained = get_embedding_matrices(handler.model)
#     del handler
#     metrics_trained = get_embedding_metrics(inst_embd_trained, data_embd_trained)
#     print("Evaluated trained embeddings")

#     print("Starting evaluation before-after inst embeddings")
#     metrics_before_after_inst = get_embedding_metrics(inst_embd_init, inst_embd_trained)
#     print("Evaluated before-after inst embeddings")

#     print("Starting evaluation before-after data embeddings")
#     metrics_before_after_data = get_embedding_metrics(data_embd_init, data_embd_trained)
#     print("Evaluated before-after data embeddings")

#     print("Starting evaluation before-after full embeddings")
#     full_embd_init = torch.cat([inst_embd_init, data_embd_init], dim=0)
#     full_embd_trained = torch.cat([inst_embd_trained, data_embd_trained], dim=0)
#     metrics_before_after_full = get_embedding_metrics(full_embd_init, full_embd_trained)
#     print("Evaluated before-after full embeddings")

#     with open(save_path, "w+") as f:
#         json.dump(
#             {
#                 "metrics_inst/data_init": metrics_init,
#                 "metrics_inst/data_trained": metrics_trained,
#                 "metrics_before_after_inst": metrics_before_after_inst,
#                 "metrics_before_after_data": metrics_before_after_data,
#                 "metrics_before_after_full": metrics_before_after_full,
#             },
#             f,
#             indent=2,
#         )

