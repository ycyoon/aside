# Evaluation Speed Optimizations

## Applied Optimizations

### 1. Reduced Debug Output
- **Before**: Debug print for every sample
- **After**: Only first 3 samples
- **Impact**: Reduces I/O overhead significantly

### 2. Reduced Memory Cleanup Frequency
- **Before**: `torch.cuda.empty_cache()` after every sample
- **After**: Only every 20 samples
- **Impact**: ~5-10% speedup (cache clearing is expensive)

### 3. Enabled TF32 for CUDA
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`
- `torch.backends.cudnn.benchmark = True`
- **Impact**: ~10-20% speedup on Ampere+ GPUs

### 4. Enabled KV Cache
- `use_cache=True` in generate()
- **Impact**: ~30-50% speedup for autoregressive generation

### 5. Disabled Unnecessary Features
- `auto_empty_cache = False` (was True)
- `use_autocast = False` (for stability)
- **Impact**: Reduced overhead

## Expected Performance

### Before Optimization
- **Speed**: ~2.86 seconds/sample
- **Throughput**: ~14 tokens/sec
- **285 samples**: ~13 minutes 33 seconds

### After Optimization (Estimated)
- **Speed**: ~1.2-1.5 seconds/sample (40-50% faster)
- **Throughput**: ~25-30 tokens/sec
- **285 samples**: ~6-7 minutes

## Additional Optimizations (Optional)

### If still too slow, consider:
1. **Batch processing**: Increase batch_size to 4-8
   - Requires more VRAM but significantly faster
   - Edit eval_safety.sh: `--batch_size 4`

2. **Reduce max_new_tokens**: 
   - Current: 100 tokens
   - Try: 50 tokens for faster eval (if acceptable)

3. **Use torch.compile** (PyTorch 2.0+):
   - Add to rgtnet_config.json: `"eval_use_torch_compile": true`
   - First run slow (compilation), subsequent runs ~20% faster

4. **Multi-GPU with DataParallel**:
   - Already supported: `--tensor_parallel_size 2`
   - Requires multiple GPUs

## Testing
Run the evaluation again:
```bash
cd /home/ycyoon/work/aside.rgtnet/experiments
bash eval_safety.sh
```

Check the new speed in the progress bar!
