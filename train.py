import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from model import GPTConfig, AvataRLLightning
from datamodule import GPTDataModule
from prepare_assets import prepare_critic_model

# try:
#     from kaggle_secrets import UserSecretsClient
#     user_secrets = UserSecretsClient()
#     wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
#     import wandb
#     wandb.login(key=wandb_api_key)
# except ImportError:
#     print("Not running on Kaggle, skipping wandb login")

# ----------------------------
# Utilities: runtime detection
# ----------------------------
def in_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

def detect_runtime() -> str:
    env = os.environ
    if any(k in env for k in ("COLAB_TPU_ADDR", "XRT_TPU_CONFIG", "TPU_NAME")):
        return "TPU"
    try:
        import torch
        if torch.cuda.is_available():
            return "GPU"
    except ImportError:
        pass
    return "CPU"

def gpu_supports_bf16() -> bool:
    import torch
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported"):
            return torch.cuda.is_bf16_supported()
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8 # Ampere or newer
    return False

def resolve_hardware_and_settings(args):
    runtime = detect_runtime()
    kaggle = in_kaggle()
    
    def pick_devices(default_count: int) -> int:
        return args.devices if args.devices > 0 else default_count

    accelerator, strategy, precision, devices = "auto", None, args.precision, 1

    if runtime == "TPU":
        import torch_xla.core.xla_model as xm
        devices = pick_devices(xm.xrt_world_size())
        accelerator, strategy = "tpu", "xla" if devices > 1 else None
        if precision == "auto": precision = "bf16-true"
    elif runtime == "GPU":
        import torch
        devices = pick_devices(torch.cuda.device_count())
        accelerator = "gpu"
        if devices > 1:
            strategy = "ddp_spawn" if kaggle else "ddp"
        if precision == "auto":
            precision = "bf16-mixed" if gpu_supports_bf16() else "16-mixed"
    else: # CPU
        accelerator = "cpu"
        if precision == "auto": precision = "32-true"

    return {
        "runtime": runtime, "kaggle": kaggle, "accelerator": accelerator,
        "strategy": strategy, "precision": precision, "devices": devices
    }

def main():
    parser = argparse.ArgumentParser(description="Train GPT with AvataRL using PyTorch Lightning")

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/openwebtext')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=1024)

    # Model arguments
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', type=bool, default=True)

    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help="Simulate larger batch sizes by accumulating gradients.")

    # AvataRL arguments
    parser.add_argument('--critic_model_path', type=str, default='out/ckpt_critic_30M.pt')
    parser.add_argument('--use_4bit_critic', action='store_true')
    parser.add_argument('--reality_weight', type=float, default=0.7)
    parser.add_argument('--mentor_weight', type=float, default=0.3)
    parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1)
    parser.add_argument('--reward_scale', type=float, default=100.0)
    parser.add_argument('--top_k', type=int, default=16)
    parser.add_argument('--entropy_coefficient', type=float, default=0.01)
    parser.add_argument('--max_reward_clamp', type=float, default=1.5)

    # Hardware arguments
    parser.add_argument('--devices', type=int, default=0, help="Number of devices to use. 0 means auto.")
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--precision', type=str, default="auto", help="auto, 32-true, 16-mixed, bf16-mixed, bf16-true, etc.")
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=0)

    # Other arguments
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default='nanogpt-avatarl')
    parser.add_argument('--val_check_interval', type=int, default=200)
    parser.add_argument('--limit_val_batches', type=int, default=80)
    parser.add_argument('--no_compile', action='store_true', help="Disable torch.compile for the model.")

    args = parser.parse_args()
    
    # --- FIX: Moved hw definition before it is used ---
    hw = resolve_hardware_and_settings(args)

    if args.num_workers == 0 and hw["runtime"] != "TPU": # TPUs have their own data loading patterns
        available_cores = os.cpu_count()
        args.num_workers = min(8, available_cores) 
        print(f"INFO: Auto-setting num_workers to {args.num_workers}. This can be overridden.")

    # Prepare assets and set seed
    prepare_critic_model(args.critic_model_path)
    pl.seed_everything(args.seed)

    # Initialize DataModule
    dm = GPTDataModule(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        runtime=hw["runtime"]
    )

    # Initialize Model Config
    gpt_config = GPTConfig(
        block_size=args.block_size,
        vocab_size=50304,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias
    )
    
    model = AvataRLLightning(
        config=gpt_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        use_scheduler=True,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        critic_model_path=args.critic_model_path,
        use_4bit_critic=args.use_4bit_critic,
        reality_weight=args.reality_weight,
        mentor_weight=args.mentor_weight,
        label_smoothing_epsilon=args.label_smoothing_epsilon,
        reward_scale=args.reward_scale,
        top_k=args.top_k,
        entropy_coefficient=args.entropy_coefficient,
        max_reward_clamp=args.max_reward_clamp,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        compile_model=(not args.no_compile)
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(dirpath='checkpoints', filename='step={step}', monitor='val/avatarl_loss', mode='min', save_top_k=3, save_last=True),
        LearningRateMonitor(logging_interval='step')
    ]

    # Logger
    logger = WandbLogger(project=args.project_name) if args.use_wandb else None

    # Trainer arguments
    trainer_kwargs = dict(
        devices=hw["devices"],
        num_nodes=args.num_nodes,
        accelerator=hw["accelerator"],
        precision=hw["precision"],
        max_steps=args.max_steps,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
    )
    if hw["strategy"]:
        trainer_kwargs['strategy'] = hw["strategy"]
    
    # Print summary
    print(f"Runtime={hw['runtime']} | accelerator={hw['accelerator']} | strategy={hw['strategy']} | precision={hw['precision']} | devices={hw['devices']}")
    
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()