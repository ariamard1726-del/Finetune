import os
import yaml
import torch
import glob
import sys
import shutil
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def main():
    config = yaml.safe_load(open("config/training_config.yaml", "r"))
    
    chunk_files = sorted(glob.glob("data/shards/legal_chunk_*.jsonl"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not chunk_files:
        print("⚠️ No chunk files found. Exiting.")
        return

    current_chunk = chunk_files[0]
    print(f"--- 🎯 Training on chunk: {current_chunk} ---")

    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'], 
        torch_dtype=torch.float32, 
        device_map="cpu", 
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    adapter_path = "models/lora_adapter" 
    
    # 🟢 بررسی دقیق: آیا فایل وزن اصلی (safetensors یا bin) وجود دارد؟
    weights_exist = os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")) or \
                    os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    if weights_exist:
        try:
            print("--- 🔄 Loading previous LoRA weights ---")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True, local_files_only=True)
            print("✅ Successfully loaded previous weights.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR loading LoRA: {e}")
            sys.exit(1) # توقف کامل
    else:
        # اگر فایل وزن نبود (حتی اگر کانفیگ بود)، از نو شروع میکنیم
        print("--- 🌱 Starting fresh LoRA training (No valid weights found) ---")
        model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=current_chunk, split="train")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=adapter_path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, 
        learning_rate=float(config['learning_rate']),
        num_train_epochs=1, 
        logging_steps=10, 
        save_strategy="steps",      
        save_steps=50,              
        save_total_limit=1,         
        optim="adamw_torch", 
        fp16=False, 
        bf16=False, 
        use_cpu=True, 
        report_to="none", 
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_datasets, data_collator=data_collator,
    )

    print("--- 🚀 Starting Instruction Tuning ---")
    trainer.train()
    
    print("--- 💾 Saving Final LoRA Adapters ---")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # پاک کردن چک‌پوینت‌ها
    for item in os.listdir(adapter_path):
        if item.startswith("checkpoint"):
            shutil.rmtree(os.path.join(adapter_path, item))

if __name__ == "__main__":
    main()
