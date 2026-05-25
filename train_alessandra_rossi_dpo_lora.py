from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from config import Config


DEFAULT_MODEL_PATH = Path(Config.local_chat_model_path)
DEFAULT_DPO_PATH = Path("data/individual_simulation_data/dpo_pool/dpo_Alessandra Rossi.json")
DEFAULT_OUTPUT_DIR = Path("data/individual_simulation_data/dpo_lora/alessandra_rossi_qwen2.5_1.5b")


@dataclass
class PreferenceSample:
    prompt_messages: list[dict[str, str]]
    chosen_response: str
    rejected_response: str
    meta: dict[str, Any]


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for param in self.base_layer.parameters():
            param.requires_grad = False

        weight = self.base_layer.weight
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False, device=weight.device, dtype=weight.dtype)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False, device=weight.device, dtype=weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_preference_samples(path: Path) -> list[PreferenceSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples: list[PreferenceSample] = []

    for item in raw:
        chosen = item["chosen"]
        rejected = item["rejected"]
        if len(chosen) < 2 or len(rejected) < 2:
            continue

        prompt_messages = [{"role": "system", "content": item["system"]}]
        prompt_messages.extend(chosen[:-1])
        chosen_response = str(chosen[-1]["content"]).strip()
        rejected_response = str(rejected[-1]["content"]).strip()
        if not chosen_response or not rejected_response:
            continue

        samples.append(
            PreferenceSample(
                prompt_messages=prompt_messages,
                chosen_response=chosen_response,
                rejected_response=rejected_response,
                meta=item.get("_meta", {}),
            )
        )

    if not samples:
        raise ValueError(f"No valid DPO samples found in {path}")
    return samples


def train_eval_split(samples: list[PreferenceSample], eval_ratio: float, seed: int) -> tuple[list[PreferenceSample], list[PreferenceSample]]:
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if eval_ratio <= 0:
        return shuffled, []

    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_samples = shuffled[:eval_size]
    train_samples = shuffled[eval_size:]
    if not train_samples:
        train_samples = eval_samples
        eval_samples = []
    return train_samples, eval_samples


def build_target_modules(arg: str) -> list[str]:
    return [name.strip() for name in arg.split(",") if name.strip()]


def find_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora(model: nn.Module, target_modules: list[str], r: int, alpha: int, dropout: float) -> list[str]:
    replaced: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(module_name.endswith(target) for target in target_modules):
            continue
        parent, attr_name = find_parent_module(model, module_name)
        setattr(parent, attr_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(module_name)
    if not replaced:
        raise ValueError(f"No target modules matched: {target_modules}")
    return replaced


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def encode_prompt_response(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    response: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        if overflow >= len(prompt_ids):
            prompt_ids = prompt_ids[-max(1, max_length - len(response_ids)) :]
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids
        else:
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids

        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class PreferenceDataset(Dataset):
    def __init__(self, samples: list[PreferenceSample], tokenizer: Any, max_length: int) -> None:
        self.rows: list[dict[str, Any]] = []
        for sample in samples:
            self.rows.append(
                {
                    "chosen": encode_prompt_response(
                        tokenizer,
                        sample.prompt_messages,
                        sample.chosen_response,
                        max_length=max_length,
                    ),
                    "rejected": encode_prompt_response(
                        tokenizer,
                        sample.prompt_messages,
                        sample.rejected_response,
                        max_length=max_length,
                    ),
                    "meta": sample.meta,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class PreferenceCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _pad_side(self, sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def _pad_block(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._pad_side([x["input_ids"] for x in features], self.pad_token_id),
            "attention_mask": self._pad_side([x["attention_mask"] for x in features], 0),
            "labels": self._pad_side([x["labels"] for x in features], -100),
        }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chosen = self._pad_block([item["chosen"] for item in batch])
        rejected = self._pad_block([item["rejected"] for item in batch])
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
        }


def get_autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shift_logits = logits[:, :-1, :]
    shift_input_ids = input_ids[:, 1:]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)

    token_log_probs = F.log_softmax(shift_logits, dim=-1)
    selected = token_log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
    selected = selected * mask
    return selected.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    rewards = (policy_logratios - ref_logratios).detach()
    return losses.mean(), rewards


@torch.no_grad()
def evaluate(
    policy_model: nn.Module,
    ref_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
) -> float:
    policy_model.eval()
    ref_model.eval()
    losses: list[float] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        autocast_ctx = get_autocast_context(device)
        with autocast_ctx:
            policy_chosen_logps = sequence_log_probs(
                policy_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            policy_rejected_logps = sequence_log_probs(
                policy_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
            ref_chosen_logps = sequence_log_probs(
                ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            ref_rejected_logps = sequence_log_probs(
                ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
            loss, _ = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
            )
        losses.append(float(loss.item()))

    policy_model.train()
    return sum(losses) / max(1, len(losses))


def extract_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def save_lora_artifacts(
    output_dir: Path,
    tokenizer: Any,
    policy_model: nn.Module,
    *,
    base_model_path: Path,
    dataset_path: Path,
    target_modules: list[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    args_dict: dict[str, Any],
    replaced_modules: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(extract_lora_state_dict(policy_model), output_dir / "adapter_model.pt")
    tokenizer.save_pretrained(output_dir)

    adapter_config = {
        "base_model_path": str(base_model_path),
        "dataset_path": str(dataset_path),
        "target_modules": target_modules,
        "replaced_modules": replaced_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(args_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Alessandra Rossi Qwen2.5-1.5B with DPO LoRA")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Local base model path.")
    parser.add_argument("--dpo-path", default=str(DEFAULT_DPO_PATH), help="Path to DPO dataset JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save LoRA weights.")
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated linear modules for LoRA.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta.")
    parser.add_argument("--max-length", type=int, default=1536, help="Max token length for prompt+response.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-step batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Held-out evaluation ratio.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-steps", type=int, default=5, help="Log every N optimizer steps.")
    parser.add_argument("--save-every-epoch", action="store_true", help="Also save per-epoch adapter checkpoints.")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle training data each epoch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_path = Path(args.model_path)
    dpo_path = Path(args.dpo_path)
    output_dir = Path(args.output_dir)
    target_modules = build_target_modules(args.target_modules)

    samples = load_preference_samples(dpo_path)
    train_samples, eval_samples = train_eval_split(samples, args.eval_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    replaced_modules = apply_lora(
        policy_model,
        target_modules=target_modules,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    freeze_non_lora_parameters(policy_model)
    for param in ref_model.parameters():
        param.requires_grad = False

    if not args.disable_gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        policy_model.enable_input_require_grads()
        ref_model.gradient_checkpointing_disable()

    trainable, total = count_trainable_parameters(policy_model)
    print(f"Loaded {len(samples)} samples -> train={len(train_samples)}, eval={len(eval_samples)}")
    print(f"Applied LoRA to {len(replaced_modules)} modules")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    train_dataset = PreferenceDataset(train_samples, tokenizer, max_length=args.max_length)
    eval_dataset = PreferenceDataset(eval_samples, tokenizer, max_length=args.max_length) if eval_samples else None
    collator = PreferenceCollator(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=collator,
    )
    eval_loader = (
        DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        if eval_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(
        [param for param in policy_model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_optimizer_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    device = next(policy_model.parameters()).device
    policy_model.train()
    ref_model.eval()
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            autocast_ctx = get_autocast_context(device)

            with autocast_ctx:
                policy_chosen_logps = sequence_log_probs(
                    policy_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                policy_rejected_logps = sequence_log_probs(
                    policy_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )
                with torch.no_grad():
                    ref_chosen_logps = sequence_log_probs(
                        ref_model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["chosen_labels"],
                    )
                    ref_rejected_logps = sequence_log_probs(
                        ref_model,
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        batch["rejected_labels"],
                    )
                loss, rewards = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=args.beta,
                )
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            running_loss += float(loss.item()) * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_steps == 0:
                    mean_reward = float(rewards.mean().item())
                    print(
                        f"epoch={epoch} step={global_step}/{total_optimizer_steps} "
                        f"loss={running_loss / max(1, step):.4f} reward_margin={mean_reward:.4f}"
                    )

        eval_loss = None
        if eval_loader is not None:
            eval_loss = evaluate(
                policy_model,
                ref_model,
                eval_loader,
                device=device,
                beta=args.beta,
            )
            print(f"epoch={epoch} eval_loss={eval_loss:.4f}")

        if args.save_every_epoch:
            epoch_dir = output_dir / f"checkpoint-epoch-{epoch}"
            save_lora_artifacts(
                epoch_dir,
                tokenizer,
                policy_model,
                base_model_path=model_path,
                dataset_path=dpo_path,
                target_modules=target_modules,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                args_dict=vars(args),
                replaced_modules=replaced_modules,
            )

    save_lora_artifacts(
        output_dir,
        tokenizer,
        policy_model,
        base_model_path=model_path,
        dataset_path=dpo_path,
        target_modules=target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        args_dict=vars(args),
        replaced_modules=replaced_modules,
    )
    print(f"Saved LoRA adapter to: {output_dir}")


if __name__ == "__main__":
    main()
