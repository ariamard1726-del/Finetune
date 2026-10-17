import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config("config/training_config.yaml")
    
    print(f"--- Loading specialized legal model: {config['base_model']} ---")
    
    # 1. توکنایزر
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. تنظیمات کوانتایز 4 بیتی (حیاتی برای گیت‌هاب اکشن)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 3. لود مدل
    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # آماده سازی برای آموزش
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False # برای جلوگیری از خطای حافظه در حین آموزش

    # 4. تنظیمات LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 5. لود داده‌های تکه‌تکه شده
    print("--- Loading Dataset Shards ---")
    data_files = [os.path.join("data/shards", f) for f in os.listdir("data/shards") if f.endswith('.jsonl')]
    if not data_files:
        raise ValueError("هیچ فایل داده‌ای در data/shards پیدا نشد! ابتدا داده‌ها را تکه‌تکه کنید.")
        
    dataset = load_dataset("json", data_files=data_files, split="train")

    # تابع توکنایز کردن
    def tokenize_function(examples):
        # فرض بر این است که داده‌ها یک کلید 'text' دارند (مثلاً: "سوال: ... \n جواب: ...")
        return tokenizer(examples["text"], truncation=True, max_length=config['max_length'], padding="max_length")

    print("--- Tokenizing Dataset ---")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. تنظیمات آموزش
    training_args = TrainingArguments(
        output_dir="./models/lora_adapter",
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation'],
        learning_rate=float(config['learning_rate']),
        num_train_epochs=config['epochs'],
        logging_steps=5,
        save_strategy="epoch",
        optim="paged_adamw_8bit", # بهینه‌سازی حافظه
        fp16=True,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    # 7. شروع آموزش
    print("--- Starting Fine-Tuning ---")
    trainer.train()
    
    # 8. ذخیره وزن‌های LoRA
    print("--- Saving LoRA Adapters ---")
    model.save_pretrained("./models/lora_adapter")
    tokenizer.save_pretrained("./models/lora_adapter")
    print("Done!")

if __name__ == "__main__":
    main()
