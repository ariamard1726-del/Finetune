import glob
import json
import math
import os
import re
import shutil
import sys
import time

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

CONFIG_PATH = "config/training_config.yaml"
SHARD_GLOB = "data/shards/legal_chunk_*.jsonl"
DONE_DIR = "data/done"
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local")


def shard_number(path):
    m = re.search(r"legal_chunk_(\d+)\.jsonl$", path)
    return int(m.group(1)) if m else 10**9


def write_status(adapter_path, status, **extra):
    os.makedirs(adapter_path, exist_ok=True)
    record = {
        "status": status,
        "run_id": RUN_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    record.update(extra)

    with open(os.path.join(adapter_path, "last_run.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(adapter_path, "train_log.json")
    history = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(record)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history[-100:], f, ensure_ascii=False, indent=2)


def build_prompt(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n"


def get_end_token(tokenizer):
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
        return "<|im_end|>"
    return tokenizer.eos_token or ""


def encode_sample(tokenizer, sample, max_length, end_token):
    msgs = sample["messages"]
    prompt = build_prompt(tokenizer, msgs[:-1])
    answer = msgs[-1]["content"] + end_token
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(a_ids) < 2 or len(p_ids) + len(a_ids) > max_length:
        return None
    return {
        "input_ids": p_ids + a_ids,
        "attention_mask": [1] * (len(p_ids) + len(a_ids)),
        "labels": [-100] * len(p_ids) + a_ids,  # فقط پاسخ آموزش می‌بیند
    }


def load_items(path, tokenizer, max_length, end_token, limit=None):
    items, dropped = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            enc = encode_sample(tokenizer, json.loads(line), max_length, end_token)
            if enc is None:
                dropped += 1
            else:
                items.append(enc)
            if limit and len(items) >= limit:
                break
    return items, dropped


def eval_loss(model, items):
    model.eval()
    losses = []
    with torch.no_grad():
        for it in items:
            ids = torch.tensor([it["input_ids"]])
            labels = torch.tensor([it["labels"]])
            out = model(
                input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels
            )
            losses.append(out.loss.item())
    return sum(losses) / len(losses) if losses else None


class TimeLimit(TrainerCallback):
    def __init__(self, minutes):
        self.deadline = time.time() + minutes * 60
        self.hit = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() > self.deadline:
            control.should_training_stop = True
            self.hit = True
        return control


def main():
    start = time.time()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    adapter_path = cfg.get("adapter_path", "models/lora_adapter")
    max_length = int(cfg.get("max_length", 1024))
    accum = int(cfg.get("gradient_accumulation", 8))
    epochs = int(cfg.get("epochs", 1))
    max_steps_cap = int(cfg.get("max_steps_per_chunk", 12))
    max_minutes = float(cfg.get("max_minutes", 270))

    chunk_files = sorted(glob.glob(SHARD_GLOB), key=shard_number)
    if not chunk_files:
        print("⚠️ No chunk files found.")
        write_status(adapter_path, "no_data")
        return

    chunk = chunk_files[0]
    print(f"--- 🎯 Training on chunk: {chunk} ---")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    end_token = get_end_token(tokenizer)

    items, dropped = load_items(chunk, tokenizer, max_length, end_token)
    print(f"samples used: {len(items)} | dropped (too long/short): {dropped}")
    if dropped > len(items):
        print("⚠️ Many samples are too long. Consider raising max_length in config.")

    os.makedirs(DONE_DIR, exist_ok=True)

    if len(items) < 4:
        print("⚠️ Too few usable samples; skipping this chunk.")
        write_status(adapter_path, "skipped_empty", chunk=os.path.basename(chunk),
                     samples=len(items), dropped=dropped)
        shutil.move(chunk, os.path.join(DONE_DIR, os.path.basename(chunk)))
        return

    val_items = []
    val_file = cfg.get("val_file")
    if val_file and os.path.exists(val_file):
        val_items, _ = load_items(
            val_file, tokenizer, max_length, end_token,
            limit=int(cfg.get("val_samples", 10)),
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.float32, device_map="cpu"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    weights_exist = os.path.exists(
        os.path.join(adapter_path, "adapter_model.safetensors")
    ) or os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))

    if weights_exist:
        print("--- 🔄 Loading previous LoRA weights ---")
        try:
            model = PeftModel.from_pretrained(
                model, adapter_path, is_trainable=True, local_files_only=True
            )
        except Exception as e:
            print(f"❌ Could not load previous adapter: {e}")
            sys.exit(1)
    else:
        print("--- 🌱 Starting fresh LoRA training ---")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    steps = min(max_steps_cap, math.ceil(len(items) / accum) * epochs)
    print(f"optimizer steps this run: {steps}")

    args = TrainingArguments(
        output_dir="outputs/tmp",
        per_device_train_batch_size=int(cfg.get("per_device_batch_size", 1)),
        gradient_accumulation_steps=accum,
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        num_train_epochs=epochs,
        max_steps=steps,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=2,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        use_cpu=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    limiter = TimeLimit(max_minutes)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(items),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
        callbacks=[limiter],
    )

    print("--- 🚀 Starting Instruction Tuning ---")
    result = trainer.train()
    train_loss = float(result.training_loss)

    if math.isnan(train_loss) or math.isinf(train_loss):
        print("❌ Training loss is NaN/inf. Not saving.")
        sys.exit(1)

    print("--- 💾 Saving LoRA adapter ---")
    model.save_pretrained(adapter_path)

    val_loss = None
    if val_items:
        print("--- 📏 Evaluating on validation samples ---")
        val_loss = eval_loss(model, val_items)
        print(f"val_loss: {val_loss}")

    status = "partial" if limiter.hit else "ok"
    write_status(
        adapter_path,
        status,
        chunk=os.path.basename(chunk),
        samples=len(items),
        dropped=dropped,
        steps=int(trainer.state.global_step),
        train_loss=round(train_loss, 4),
        val_loss=None if val_loss is None else round(val_loss, 4),
        minutes=round((time.time() - start) / 60, 1),
        time_limit_hit=limiter.hit,
    )

    shutil.move(chunk, os.path.join(DONE_DIR, os.path.basename(chunk)))
    print(f"✅ Done ({status}). Chunk moved to {DONE_DIR}/")


if __name__ == "__main__":
    main()
