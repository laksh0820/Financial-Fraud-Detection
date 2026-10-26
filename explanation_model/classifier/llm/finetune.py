"""LoRA fine-tuning of the LLM pattern classifier on train.jsonl (prompt -> "- Observed Pattern: X")."""
import gc
import logging
import os

from common.utils import read_jsonl
from explanation_model.classifier.llm.modeling import load_base_model, load_tokenizer, pick_dtype
from explanation_model.classifier.llm.prompts import render_prompt


def tokenize_examples(examples, tokenizer, max_seq_len):
    out, n_dropped = [], 0
    for ex in examples:
        prompt_ids = tokenizer(render_prompt(tokenizer, ex["prompt"]), add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(ex["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + completion_ids
        if len(input_ids) > max_seq_len:
            n_dropped += 1
            continue
        out.append({
            "input_ids": input_ids,
            "labels": [-100] * len(prompt_ids) + completion_ids,  # loss on the answer only
            "attention_mask": [1] * len(input_ids),
        })
    if n_dropped:
        logging.warning(f"Dropped {n_dropped}/{len(examples)} training examples longer than "
                        f"{max_seq_len} tokens (lower --max_block_edges or raise --max_seq_len).")
    return out


def collate_fn(batch, pad_id):
    import torch
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        ids.append([pad_id] * pad + b["input_ids"])
        labels.append([-100] * pad + b["labels"])
        mask.append([0] * pad + b["attention_mask"])
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}


def train(args, paths):
    import torch
    from torch.utils.data import Dataset
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    dtype, use_bf16 = pick_dtype()
    tokenizer = load_tokenizer(args.base_model)

    examples = read_jsonl(paths["train"])
    data = tokenize_examples(examples, tokenizer, args.max_seq_len)
    logging.info(f"{len(data)} tokenized training examples "
                 f"(mean length {sum(len(d['input_ids']) for d in data) / max(1, len(data)):.0f} tokens)")

    class ListDataset(Dataset):
        def __len__(self): return len(data)
        def __getitem__(self, i): return data[i]

    model = load_base_model(args.base_model, args.load_in_4bit, dtype)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()  # needed for gradient checkpointing + LoRA

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules, bias="none", task_type="CAUSAL_LM",
    ))
    if not use_bf16:  # fp16 AMP needs fp32 trainable params
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.work_dir, "trainer_out"),
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",           # only the final adapter is kept
        eval_strategy="no",           # test split is never touched during training
        bf16=use_bf16, fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=ListDataset(),
        data_collator=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    logging.info("Starting fine-tuning ...")
    trainer.train()

    logging.info(f"Saving LoRA adapter to {paths['adapter']}")
    model.save_pretrained(paths["adapter"])
    tokenizer.save_pretrained(paths["adapter"])

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


