import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_name = "alphaedge-ai/Qwen3-1.7B-fas-32768"
    # 🟢 استفاده از مسیر نسبی به جای مسیر مطلق برای رفع باگ
    adapter_path = "models/lora_adapter"
    
    print(f"--- Loading Base Model: {base_model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float32, 
        device_map="cpu", 
        trust_remote_code=True
    )

    # بررسی وجود آداپتور
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print("--- Loading LoRA Adapter ---")
        try:
            # استفاده از مسیر نسبی
            model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
            print("✅ Adapter loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading adapter: {e}")
            print("Testing base model only.")
    else:
        print("⚠️ No adapter found! Testing base model only.")

    # پرامپت تست
    system_prompt = "تو یک دستیار حقوقی هستی. به سوال کاربر با استدلال منطقی پاسخ بده."
    user_prompt = "من یک پرونده در گروه حقوقی دارم. اگر قرارداد شفاهی باشد، آیا قابل پیگیری در دادگاه است؟"

    # فرمت‌دهی استاندارد Qwen
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to("cpu")

    print("\n--- Generating Response (Please wait) ---")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=0.7, 
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print("\n================ مدل پاسخ داد ================")
    print(response)
    print("==============================================")

if __name__ == "__main__":
    main()
