import math
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

from utils import compute_avatarl_loss, load_critic_model

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    """Plain PyTorch GPT model. Forward returns logits only."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        return logits

    def crop_block_size(self, block_size: int):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[Dict[str, Any]] = None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)

        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))
        ]
        transposed = [
            "attn.c_attn.weight", "attn.c_proj.weight",
            "mlp.c_fc.weight", "mlp.c_proj.weight",
        ]
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class GPTLightning(pl.LightningModule):
    """Base Lightning wrapper around GPT."""

    def __init__(
        self,
        config: GPTConfig
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT(config)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        return self.model(idx)

    @staticmethod
    def _unpack_batch(
        batch: Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.LongTensor, Optional[torch.LongTensor], Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            idx = (
                batch.get("input_ids") or batch.get("idx") or
                batch.get("inputs") or batch.get("x")
            )
            targets = batch.get("labels") or batch.get("targets") or batch.get("y")
            mask = batch.get("attention_mask") or batch.get("mask")
        elif isinstance(batch, (tuple, list)):
            idx = batch[0]
            targets = batch[1] if len(batch) > 1 else None
            mask = batch[2] if len(batch) > 2 else None
        else:
            idx = batch
            targets = None
            mask = None

        if idx is None:
            raise ValueError("Batch does not contain input_ids/idx")
        return idx.long(), (targets.long() if targets is not None else None), mask

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, V = logits.shape
        per_tok = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=-1,
            reduction="none",
        ).view(B, T)

        if mask is not None:
            mask = mask.float()
            denom = mask.sum().clamp_min(1.0)
            loss = (per_tok * mask).sum() / denom
        else:
            valid = (targets != -1).float()
            denom = valid.sum().clamp_min(1.0)
            loss = (per_tok * valid).sum() / denom
        return loss

    def training_step(self, batch, batch_idx):
        idx, targets, mask = self._unpack_batch(batch)
        if targets is None:
            raise RuntimeError("No targets provided to compute training loss.")
        logits = self(idx)
        loss = self.compute_loss(logits, targets, mask)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, targets, mask = self._unpack_batch(batch)
        if targets is None:
            return None
        logits = self(idx)
        loss = self.compute_loss(logits, targets, mask)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=idx.size(0))
        self.log("val/ppl", torch.exp(loss.detach()), prog_bar=False, on_step=False, on_epoch=True, batch_size=idx.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        idx, targets, mask = self._unpack_batch(batch)
        if targets is None:
            return None
        logits = self(idx)
        loss = self.compute_loss(logits, targets, mask)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=idx.size(0))
        self.log("test/ppl", torch.exp(loss.detach()), prog_bar=False, on_step=False, on_epoch=True, batch_size=idx.size(0))
        return loss

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        if not self.hparams.use_scheduler:
            return optimizer

        if self.hparams.max_steps is None:
            print("WARNING: use_scheduler=True but max_steps=None. Returning optimizer without scheduler.")
            return optimizer

        def lr_lambda(current_step: int):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            progress = (current_step - self.hparams.warmup_steps) / float(
                max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def crop_block_size(self, block_size: int):
        return self.model.crop_block_size(block_size)

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[Dict[str, Any]] = None, **lightning_kwargs):
        base = GPT.from_pretrained(model_type, override_args)
        lm = cls(base.config, **lightning_kwargs)
        lm.model.load_state_dict(base.state_dict())
        return lm


class AvataRLLightning(GPTLightning):
    """AvataRL extension of GPTLightning with critic-based reward modeling."""

    def __init__(
        self,
        config: GPTConfig,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        use_scheduler: bool = True,
        warmup_steps: int = 200,
        max_steps: Optional[int] = 100000,
        critic_model_path: str = None,
        use_4bit_critic: bool = False
    ):
        super().__init__(config, learning_rate, weight_decay, betas, use_scheduler, warmup_steps, max_steps)
        self.save_hyperparameters()
        self.automatic_optimization = False

        if critic_model_path:
            self.critic_model = load_critic_model(critic_model_path, use_4bit_critic)
            self.critic_model.eval()
            for param in self.critic_model.parameters():
                param.requires_grad = False
        else:
            raise ValueError("critic_model_path must be provided for AvataRL training")

    def setup(self, stage: str):
        if self.hparams.compile_model:
            backend = 'openxla' if self.device.type == 'xla' else 'inductor'
            
            print(f"INFO: Compiling `self.model` with torch.compile(backend='{backend}') on rank {self.global_rank}...")
            self.model = torch.compile(self.model, backend=backend)
            
            print(f"INFO: Compiling `self.critic_model` with torch.compile(backend='{backend}') on rank {self.global_rank}...")
            self.critic_model = torch.compile(self.critic_model, backend=backend)

    def on_train_start(self):
        if hasattr(self, 'critic_model'):
            self.critic_model = self.critic_model.to(self.device)

    def training_step(self, batch, batch_idx):
        # --- Forward pass ---
        idx, targets, mask = self._unpack_batch(batch)
        student_logits = self(idx)
        with torch.no_grad():
            critic_logits = self.critic_model(idx)
        
        loss, metrics = compute_avatarl_loss(
            student_logits, critic_logits, targets,
            reality_weight=self.hparams.reality_weight,
            mentor_weight=self.hparams.mentor_weight,
            label_smoothing_epsilon=self.hparams.label_smoothing_epsilon,
            reward_scale=self.hparams.reward_scale,
            top_k=self.hparams.top_k,
            entropy_coefficient=self.hparams.entropy_coefficient,
            max_reward_clamp=self.hparams.max_reward_clamp
        )
        
        self.log("train/avatarl_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        for k, v in metrics.items():
            self.log(f"metrics/{k}", v, on_step=False, on_epoch=True, sync_dist=True)

        scaled_loss = loss / self.hparams.accumulate_grad_batches
        self.manual_backward(scaled_loss)
        
        if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
            opt = self.optimizers()
            sch = self.lr_schedulers()

            if hasattr(self.trainer.precision_plugin, "scaler") and self.trainer.precision_plugin.scaler:
                self.trainer.precision_plugin.scaler.unscale_(opt)
            
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.hparams.gradient_clip_val
            )

            opt.step()
            opt.zero_grad()
            sch.step()

            self.log("lr", sch.get_last_lr()[0], prog_bar=False, on_step=True, on_epoch=False)


    def validation_step(self, batch, batch_idx):
        idx, targets, mask = self._unpack_batch(batch)
        if targets is None: return None
        student_logits = self(idx)
        with torch.no_grad():
            critic_logits = self.critic_model(idx)

        loss, metrics = compute_avatarl_loss(
            student_logits, critic_logits, targets,
            reality_weight=self.hparams.reality_weight,
            mentor_weight=self.hparams.mentor_weight,
            label_smoothing_epsilon=self.hparams.label_smoothing_epsilon,
            reward_scale=self.hparams.reward_scale,
            top_k=self.hparams.top_k,
            entropy_coefficient=self.hparams.entropy_coefficient,
            max_reward_clamp=self.hparams.max_reward_clamp
        )
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1), ignore_index=-1
        )
        self.log("val/avatarl_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/ce_loss", ce_loss, prog_bar=False, on_epoch=True, sync_dist=True)
        for k, v in metrics.items():
            self.log(f"val_metrics/{k}", v, on_epoch=True, sync_dist=True)
        return loss