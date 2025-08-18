#!/usr/bin/env python3
"""RGTNet Smoke Test with Performance Metrics

Quick sanity check for unified RGTNet loading + batch generation path.
Runs a few short prompts through the evaluation stack WITHOUT external datasets.

Usage (example):
  python experiments/smoke_test_rgtnet.py \
      --model_dir experiments/models/rgtnet_qwen2.5-7b_20250814_0759/merged_epoch_1 \
      --embedding_type rgtnet \
      --batch_size 4 \
      --max_new_tokens 32

Exit code 0 on success, >0 on failure.
"""
import argparse
import sys
import time
import torch
import os
from model_api import CustomModelHandler

# Performance optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable Flash Attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass

PROMPTS = [
    ("You are a concise assistant.", "List three prime numbers."),
    ("You are a concise assistant.", "Explain what a transformer model is in one sentence."),
    ("You are a concise assistant.", "What is 7 * 13?"),
    ("You are a concise assistant.", "Summarize the importance of data privacy in 10 words."),
    ("You are a concise assistant.", "What is the capital of France?"),
    ("You are a concise assistant.", "Calculate 15 + 27."),
    ("You are a concise assistant.", "Name a popular programming language."),
    ("You are a concise assistant.", "What year was Python created?"),
]


def run_smoke(args):
    if args.embedding_type not in ("rgtnet", "rgtnet_orthonly"):
        print(f"[WARN] embedding_type {args.embedding_type} is not RGTNet variant; continuing anyway.")

    # Enable GPU optimizations early
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        print(f"[GPU] Optimizations enabled: TF32=True, Flash_SDPA=attempted")

    handler = CustomModelHandler(
        checkpoint_path=args.model_dir,
        instruct_model_path=args.model_dir,  # base_model param reused for size inference if needed
        data_model_path=args.model_dir,
        tokenizer_path=args.model_dir,
        chat_template_path=None,
        prompt_ix=0,
        embedding_type=args.embedding_type,
        load_from_checkpoint=True,
        model_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        max_token_len=args.max_input_len,
    )

    # Build batched inputs (truncate to batch_size if requested smaller than PROMPTS length)
    use_prompts = PROMPTS[: args.batch_size]
    system_instructions = [s for s, _ in use_prompts]
    user_instructions = [u for _, u in use_prompts]

    print(f"[Smoke] Running batch of {len(use_prompts)} prompts | embedding_type={args.embedding_type}")

    # Warmup run to avoid cold start effects
    print("[Smoke] Warmup run...")
    try:
        _, _ = handler.call_model_api_batch(
            system_instructions[:1],
            user_instructions[:1], 
            max_new_tokens=4,
            do_sample=False,
        )
    except Exception as e:
        print(f"[WARN] Warmup failed: {e}")

    # Actual timed run
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        gpu_start = time.time()
    
    try:
        responses, _ = handler.call_model_api_batch(
            system_instructions,
            user_instructions,
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"[FAIL] Exception during batch generation: {e}")
        raise
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for GPU operations to complete
    total_time = time.time() - start_time
    gpu_time = time.time() - gpu_start if torch.cuda.is_available() else total_time

    # Performance metrics
    gen_token_counts = []
    approx_input_tokens = []
    for r in responses:
        # Rough: tokenize again to count output tokens
        tokenized = handler.tokenizer(r, add_special_tokens=False).input_ids
        gen_token_counts.append(len(tokenized))
    for u in user_instructions:
        tokenized_in = handler.tokenizer(u, add_special_tokens=False).input_ids
        approx_input_tokens.append(len(tokenized_in))
    total_gen = sum(gen_token_counts)
    total_inp = sum(approx_input_tokens)
    toks_per_sec = total_gen / total_time if total_time > 0 else 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
    else:
        mem_alloc = mem_reserved = 0.0

    for i, (inp, resp) in enumerate(zip(user_instructions, responses), 1):
        print(f"\n--- Sample {i} ---")
        print(f"User: {inp}")
        print(f"Response: {resp}")
        if len(resp.strip()) == 0:
            print("[WARN] Empty response")

    # Basic assertions
    non_empty = sum(1 for r in responses if len(r.strip()) > 0)
    if non_empty != len(use_prompts):
        print(f"[FAIL] Some responses empty: {non_empty}/{len(use_prompts)} non-empty")
        return 2

    # Simple quality heuristic: each response should contain at least one alphabetic char
    bad_quality = [i for i, r in enumerate(responses) if not any(c.isalpha() for c in r)]
    if bad_quality:
        print(f"[WARN] Responses with low content heuristic: {bad_quality}")

    print("\n[PASS] RGTNet smoke test succeeded.")
    print(
        f"[METRICS] batch={len(use_prompts)} input_tokens={total_inp} generated_tokens={total_gen} total_time={total_time:.3f}s gpu_time={gpu_time:.3f}s speed={toks_per_sec:.2f} tok/s mem_alloc={mem_alloc:.1f}MB mem_reserved={mem_reserved:.1f}MB"
    )
    
    # Additional diagnostics
    if torch.cuda.is_available():
        print(f"[GPU] Device: {torch.cuda.get_device_name()}")
        print(f"[GPU] TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"[GPU] Flash SDPA: {getattr(torch.backends.cuda, 'flash_sdp_enabled', lambda: 'unknown')()}")
    
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Path to RGTNet (or wrapper) checkpoint directory")
    p.add_argument("--embedding_type", default="rgtnet", choices=["rgtnet", "rgtnet_orthonly", "single_emb"], help="Embedding type to test")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size (<= number of static prompts)")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--max_input_len", type=int, default=512)
    p.add_argument("--do_sample", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    code = run_smoke(args)
    sys.exit(code)
