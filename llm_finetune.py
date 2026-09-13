"""
Fine-tunes a local causal LLM (e.g. Qwen/Qwen3-14B) with LoRA on the
(prompt, completion) pairs produced by mine_finetune_data.py.

Dependencies:
    pip install peft bitsandbytes --break-system-packages

Usage:
    python mine_finetune_data.py --data Small_HI --model gin --unique_name run1 \
        --patterns_file HI-Small_Patterns.txt --n_samples 2000 --stratify \
        --out finetune_train.jsonl

    python finetune_llm.py --train_path finetune_train.jsonl \
        --base_model Qwen/Qwen3-14B --output_dir ./qwen3_14b_aml_lora \
        --n_epochs 3 --load_in_4bit

    # Then, at eval time:
    python evaluate_llm_reasoning.py --data Small_HI --model gin --unique_name run1 \
        --n_samples 200 --stratify --max_subgraph_edges 20 \
        --llm_model Qwen/Qwen3-14B --adapter_path ./qwen3_14b_aml_lora \
        --out_path llm_eval_results.csv
"""

import argparse
import json
import logging
import random

import torch
from torch.utils.data import Dataset

from util import logger_setup, set_seed


def _load_system_prompt():
    # Imported lazily so this script doesn't need torch_geometric / the GNN
    # stack to be importable just to read a string constant.
    from llm_reasoning import SYSTEM_PROMPT
    return SYSTEM_PROMPT


class PromptCompletionDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_len=4096, enable_thinking=None):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking
        self.system_prompt = _load_system_prompt()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ex["prompt"]},
        ]

        template_kwargs = dict(add_generation_prompt=True, tokenize=True)
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking

        prompt_ids = self.tokenizer.apply_chat_template(prompt_messages, **template_kwargs)
        completion_ids = self.tokenizer(
            ex["completion"] + self.tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        if len(input_ids) > self.max_seq_len:
            overflow = len(input_ids) - self.max_seq_len
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


def collate_fn(batch, pad_token_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append([pad_token_id] * pad_len + b["input_ids"])
        labels.append([-100] * pad_len + b["labels"])
        attention_mask.append([0] * pad_len + b["attention_mask"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def load_examples(path):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="finetune_train.jsonl",
                        help="JSONL produced by mine_finetune_data.py (needs 'prompt' and "
                             "'completion' fields).")
    parser.add_argument("--val_frac", type=float, default=0.05,
                        help="Fraction of --train_path held out for eval loss during training. "
                             "Set to 0 to disable.")
    parser.add_argument("--base_model", default="Qwen/Qwen3-14B")
    parser.add_argument("--output_dir", default="./llm_aml_lora")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs='+',
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                        help="Attention/MLP projection names to attach LoRA adapters to. The "
                             "defaults match Qwen3's module names.")
    parser.add_argument("--load_in_4bit", action='store_true',
                        help="QLoRA-style 4-bit base model")
    parser.add_argument("--disable_thinking_in_training", action='store_true',
                        help="Pass enable_thinking=False when building the training chat "
                             "template. Set this to match --llm_disable_thinking at eval time. "
                             "If not set (default), training matches normal thinking-enabled "
                             "inference (an empty <think></think> block ahead of the answer).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    return parser


def main():
    args = create_parser().parse_args()
    logger_setup()
    set_seed(args.seed)
    random.seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logging.info(f"Loading tokenizer for {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(device_map="auto", attn_implementation="sdpa")
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    logging.info(f"Loading base model {args.base_model} "
                 f"(first run downloads from the Hub, this can take a while) ...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    model.config.use_cache = False  # required alongside gradient checkpointing

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logging.info(f"Loading training examples from {args.train_path} ...")
    examples = load_examples(args.train_path)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac)) if args.val_frac > 0 else 0
    val_examples, train_examples = examples[:n_val], examples[n_val:]
    logging.info(f"{len(train_examples)} train examples, {len(val_examples)} eval examples.")

    enable_thinking = False if args.disable_thinking_in_training else None
    train_dataset = PromptCompletionDataset(train_examples, tokenizer, args.max_seq_len, enable_thinking)
    eval_dataset = (
        PromptCompletionDataset(val_examples, tokenizer, args.max_seq_len, enable_thinking)
        if val_examples else None
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        report_to=[],
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    logging.info("Starting fine-tuning ...")
    trainer.train()

    logging.info(f"Saving LoRA adapter to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("Done. Pass --adapter_path "
                 f"{args.output_dir} (and --llm_model {args.base_model}) to "
                 "evaluate_llm_reasoning.py or llm_reasoning.py to use it.")


if __name__ == "__main__":
    main()