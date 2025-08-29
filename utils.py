import torch
from torch import Tensor
from torch.nn import functional as F

def compute_avatarl_loss(
    student_logits: Tensor,
    critic_logits: Tensor,
    ground_truth_tokens: Tensor,
    reality_weight: float = 0.7,
    mentor_weight: float = 0.3,
    label_smoothing_epsilon: float = 0.1,
    reward_scale: float = 100.0,
    top_k: int = 16,
    entropy_coefficient: float = 0.01,
    max_reward_clamp: float = 1.5
) -> tuple[Tensor, dict]:

    B, T, V = student_logits.shape
    N = B * T
    student_logits_flat = student_logits.view(N, V)
    critic_logits_flat = critic_logits.view(N, V)
    gt_flat = ground_truth_tokens.view(N)

    # --- 1. Construct a fixed-size action set for each token position ---
    # Combine top-k tokens from student, critic, and the ground-truth token.
    _, student_topk = student_logits_flat.topk(top_k, dim=-1)
    _, critic_topk = critic_logits_flat.topk(top_k, dim=-1)
    combined_indices = torch.cat([student_topk, critic_topk, gt_flat.unsqueeze(1)], dim=1)

    # Getting unique actions without changing tensor shape to be TPU friendly
    sorted_vals, sort_idx = combined_indices.sort(dim=1)
    is_first = torch.cat(
        [torch.ones(N, 1, dtype=torch.bool, device=combined_indices.device),
         (sorted_vals[:, 1:] != sorted_vals[:, :-1])],
        dim=1
    )
    unsort_idx = sort_idx.argsort(dim=1)
    unique_mask = is_first.gather(1, unsort_idx)  # Mask of unique actions
    action_indices = combined_indices            
    action_masks = unique_mask                  

    # --- 2. Create the ideal probability distribution (P_ideal) ---
    # Mentor distribution (from the critic model)
    mentor_probs = F.softmax(critic_logits_flat, dim=-1)

    # Reality distribution (label-smoothed ground truth over the action set)
    reality_probs = torch.zeros_like(mentor_probs)
    batch_idx = torch.arange(N, device=reality_probs.device)
    reality_probs[batch_idx, gt_flat] = 1.0 - label_smoothing_epsilon # Assign high prob to ground truth
    
    # Distribute smoothing epsilon across other unique, non-GT actions
    num_active = action_masks.sum(dim=1, keepdim=True).float()
    denom = torch.clamp(num_active - 1.0, min=1.0) # Avoid division by zero
    smoothing_per_token = label_smoothing_epsilon / denom
    smoothing_map = smoothing_per_token.expand_as(action_indices) * action_masks.float()
    reality_probs.scatter_add_(1, action_indices, smoothing_map)
    reality_probs[batch_idx, gt_flat] = 1.0 - label_smoothing_epsilon # Re-assert GT prob

    # P_ideal is the product of experts: P_ideal ∝ P_reality^a * P_mentor^b
    ideal_probs = (reality_probs.clamp_min(1e-20).pow(reality_weight) *
                   mentor_probs.clamp_min(1e-20).pow(mentor_weight))

    # --- 3. Calculate rewards based on the ideal distribution ---
    # Gather ideal probabilities for our action set and normalize them
    action_probs_raw = ideal_probs.gather(1, action_indices)
    masked_action_probs = action_probs_raw * action_masks.float()
    action_probs_sum = masked_action_probs.sum(dim=1, keepdim=True)
    action_probs_norm = masked_action_probs / action_probs_sum.clamp_min(1e-8)

    # Reward is given only to actions with a probability above the average for that position
    valid_action_counts = action_masks.sum(dim=1, keepdim=True).float()
    mean_prob = action_probs_sum / valid_action_counts.clamp_min(1e-8)
    above_mean_mask = (action_probs_norm > mean_prob) & action_masks
    action_rewards = torch.where(
        above_mean_mask, action_probs_norm * reward_scale, torch.zeros_like(action_probs_norm)
    )
    action_rewards = action_rewards * action_masks.float()

    # Clamp rewards proportionally to avoid extreme values
    max_reward_per_seq = action_rewards.max(dim=1, keepdim=True).values
    scale = torch.where(
        max_reward_per_seq > max_reward_clamp,
        max_reward_clamp / max_reward_per_seq.clamp_min(1e-8),
        torch.ones_like(max_reward_per_seq)
    )
    action_rewards = action_rewards * scale

    # --- 4. Calculate the final policy gradient loss ---
    temperature = 1.0 + entropy_coefficient # Additive entropy bonus
    student_log_probs_full = F.log_softmax(student_logits_flat / temperature, dim=-1)
    student_log_probs_actions = student_log_probs_full.gather(1, action_indices) * action_masks.float()
    
    pg_loss = -(student_log_probs_actions * action_rewards.detach()).sum(dim=1)
    pg_loss = pg_loss / valid_action_counts.squeeze(1).clamp_min(1e-8)
    total_loss = pg_loss.mean()

    # --- 5. Compute metrics for logging  ---
    # removed .item() calls
    valid_rewards = action_rewards[action_masks]
    avg_reward = valid_rewards.mean() if valid_rewards.numel() > 0 else torch.tensor(0.0)
    max_reward = valid_rewards.max() if valid_rewards.numel() > 0 else torch.tensor(0.0)

    return total_loss, {
        "avg_reward": avg_reward.detach(),
        "max_reward": max_reward.detach(),
        "avg_action_space_size": valid_action_counts.mean().detach(),
    }

def load_critic_model(checkpoint_path: str, use_4bit: bool):
    from model import GPTConfig, GPT
    if use_4bit:
        raise NotImplementedError("4-bit loading not implemented in this example.")
    
    print(f"Loading critic model in FP16/BF16 from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = checkpoint["model_args"]
    state_dict = checkpoint["model"]
    
    gptconf = GPTConfig(**model_args)
    critic = GPT(gptconf)
    
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    critic.load_state_dict(state_dict)
    critic.eval()
    for param in critic.parameters():
        param.requires_grad = False
        
    print("Critic loaded successfully.")
    return critic