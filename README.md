## How to Run

The training script `train.py` automatically detects and configures settings for your hardware (GPU, TPU, CPU).

-----

### Single GPU

The script defaults to using a single GPU if available. Specify precision for optimal performance.

```bash
# For NVIDIA Ampere (A100, 30xx series) or newer GPUs
python train.py --batch_size 16 --accumulate_grad_batches 4 --precision "bf16-mixed"

# For older NVIDIA GPUs
python train.py --batch_size 16 --accumulate_grad_batches 4 --precision "16-mixed"
```

-----

### Multi-GPU (DDP)

Specify the number of devices to enable distributed training. Adjust batch sizes to maintain the global batch size.

```bash
# 4 GPUs
python train.py \
    --devices 4 \
    --strategy "ddp" \
    --batch_size 8 \
    --accumulate_grad_batches 8 \
    --precision "bf16-mixed"
```

-----

### TPU

In a TPU environment (e.g., Google Colab, Kaggle), the accelerator and strategy are configured automatically.

```bash
python train.py --batch_size 16 --accumulate_grad_batches 4
```

-----

### CPU

Fallback to CPU by default - For debugging if any.

```bash
python train.py --batch_size 4 --precision "32-true" --no_compile
```

For more options please see lightning official docs for [strategy](https://lightning.ai/docs/pytorch/stable/extensions/strategy.html) and [mixed_precision](https://lightning.ai/docs/pytorch/1.5.9/advanced/mixed_precision.html).
