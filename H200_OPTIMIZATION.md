# H200 GPU Optimization Guide

Your NVIDIA H200 has **143GB memory** - you're currently using only **3.8GB (~2.6%)**. Here's how to maximize performance:

---

## Quick Performance Commands

### Validation (Fast)
```bash
# Recommended for H200 (5-10x faster)
python validate.py --eval-all-on-dataset MeetingBank \
    --no-quantization \
    --batch-size 64

# Even faster (if stable)
python validate.py --eval-all-on-dataset MeetingBank \
    --no-quantization \
    --batch-size 128
```

### Training (Fast)
```bash
# Recommended for H200 (3-5x faster)
python training.py \
    --no-quantization \
    --batch-size 128

# Even faster (if memory allows)
python training.py \
    --no-quantization \
    --batch-size 256
```

---

## Key Optimizations

### 1. **Disable Quantization** ⚡ (BIGGEST IMPACT)

**Current:** 4-bit quantization (saves memory, slower inference)
**Recommended:** Full precision (much faster, plenty of memory available)

```bash
--no-quantization
```

**Why?**
- 4-bit quantization is for GPUs with 16-24GB memory
- Your H200 has 143GB - you don't need it
- Full precision is **2-3x faster** for inference/training

### 2. **Increase Batch Size** ⚡

**Current:**
- Validation: batch_size=4 (now 32 default)
- Training: batch_size=32

**Recommended for H200:**
- Validation: `--batch-size 64` or `--batch-size 128`
- Training: `--batch-size 128` or `--batch-size 256`

**Why?**
- Larger batches = more parallel computation
- Better GPU utilization
- Faster overall throughput

### 3. **Memory Estimates**

For OpenLLaMA 3B model on H200:

| Configuration | Memory Usage | Speed | Recommended |
|--------------|--------------|-------|-------------|
| 4-bit quant, batch=4 | ~4GB | Baseline | ❌ Too slow |
| 4-bit quant, batch=32 | ~8GB | 1.5x | ❌ Underutilized |
| Full precision, batch=32 | ~12GB | 2x | ⚠️ Better |
| Full precision, batch=64 | ~18GB | 3x | ✅ Good |
| Full precision, batch=128 | ~28GB | 5x | ✅ Optimal |
| Full precision, batch=256 | ~50GB | 7x | ✅ Maximum |

---

## Testing Performance

### 1. Test Base Model Evaluation
```bash
# Quick test on small dataset
python validate.py --eval-base-model C-STANCE \
    --no-quantization \
    --batch-size 128
```

### 2. Monitor GPU Usage
```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi
```

You should see:
- **Higher memory usage** (20-50GB instead of 3.8GB)
- **Higher GPU utilization** (80-100% instead of 13%)
- **Faster completion time**

### 3. Find Optimal Batch Size
```bash
# Try increasing until you hit memory limits
python validate.py --eval-base-model C-STANCE --no-quantization --batch-size 64
python validate.py --eval-base-model C-STANCE --no-quantization --batch-size 128
python validate.py --eval-base-model C-STANCE --no-quantization --batch-size 256
```

---

## Expected Speedup

Based on your H200 capabilities:

| Task | Current Time | Optimized Time | Speedup |
|------|--------------|----------------|---------|
| Single validation | ~10 min | ~2 min | **5x** |
| All checkpoints (8 models) | ~80 min | ~15 min | **5x** |
| Training one task | ~30 min | ~8 min | **3-4x** |
| Full continual learning | ~4 hours | ~1 hour | **4x** |

---

## Advanced Optimizations (Optional)

### 4. Use Mixed Precision Training
The training script already uses `bf16=True` (good!)

### 5. Gradient Accumulation (if batch too large)
If you hit OOM with large batches:
```python
# In training.py, modify TrainingArguments:
gradient_accumulation_steps=4  # Effective batch = batch_size * 4
per_device_train_batch_size=64  # Smaller per-device batch
```

### 6. Parallel Data Loading
Already set to 4 workers (good for validation)

---

## Recommended Workflow

### For Evaluation
```bash
# Evaluate all checkpoints + base model on a dataset (FAST)
python validate.py --eval-all-on-dataset MeetingBank \
    --no-quantization \
    --batch-size 128 \
    --checkpoint-base-dir /home/hice1/kmadaswar3/scratch/smr_project/lora-continual
```

### For Training
```bash
# Continue/resume training (FAST)
python training.py \
    --no-quantization \
    --batch-size 128 \
    --output-dir /home/hice1/kmadaswar3/scratch/smr_project/lora-continual
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 64  # instead of 128
```

### CUDA Out of Memory
```bash
# Clear GPU cache before running
python -c "import torch; torch.cuda.empty_cache()"
```

### Check Current GPU Usage
```bash
nvidia-smi
```

---

## Summary

**Current Setup:** Conservative (for 16-24GB GPUs)
**Your H200:** 143GB - massively underutilized

**Quick Win Commands:**
```bash
# Validation (5x faster)
python validate.py --eval-all-on-dataset MeetingBank --no-quantization --batch-size 128

# Training (4x faster)
python training.py --no-quantization --batch-size 128
```

**Expected Results:**
- GPU memory usage: 3.8GB → 30-50GB ✅
- GPU utilization: 13% → 80-100% ✅
- Speed: **4-5x faster** ✅
