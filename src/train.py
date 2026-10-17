import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config("config/training_config.yaml")
    
    print(f"--- Loading model: {config['base_model']} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # لود مدل با دقت 32 بیت (مناسب برای CPU) بدون bitsandbytes
    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        torch_dtype=torch.float32,
        device_map="cpu", # اجبار به استفاده از CPU
        trust_remote_code=True
    )
    
    model.config.use_cache = False

    # تنظیمات LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)

    print("--- Loading Dataset Shards ---")
    data_files = [os.path.join("data/shards", f) for f in os.listdir("data/shards") if f.endswith('.jsonl')]
    dataset = load_dataset("json", data_files=data_files, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config['max_length'], padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # تنظیمات آموزش (مخصوص CPU)
    training_args = TrainingArguments(
        output_dir="./models/lora_adapter",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1, 
        learning_rate=float(config['learning_rate']),
        num_train_epochs=1, 
        logging_steps=1, 
        save_strategy="epoch",
        optim="adamw_torch", 
        fp16=False, 
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    print("--- Starting Fine-Tuning on CPU ---")
    trainer.train()
    
    print("--- Saving LoRA Adapters ---")
    model.save_pretrained("./models/lora_adapter")
    tokenizer.save_pretrained("./models/lora_adapter")
    print("Done!")

if __name__ == "__main__":
    main()