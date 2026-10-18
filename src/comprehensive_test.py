import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_response(model, tokenizer, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.3, # دمای پایین برای جواب‌های دقیق‌تر
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    base_model_name = "alphaedge-ai/Qwen3-1.7B-fas-32768"
    adapter_path = "models/lora_adapter"
    
    print("--- Loading Model for Comprehensive Test ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )

    weights_exist = os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")) or \
                    os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))

    if weights_exist:
        print("--- ✅ LoRA Adapter Found and Loaded ---")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    else:
        print("--- ⚠️ No LoRA Adapter Found! Testing Base Model Only. ---")

    # پرامپت سیستمی که در داده‌های آموزشی استفاده کردیم
    sys_prompt = """[نقش و شخصیت]
تو یک «دستیار حقوقی ارشد، دقیق و فارسی‌زبان» هستی. وظیفه تو تحلیل مسائل حقوقی، استخراج قوانین مرتبط، استدلال منطقی و ارائه راهکارهای عملی است.
[قوانین طلایی]
۱. همیشه ساختار استدلال گام‌به‌گام داشته باش.
۲. پاسخ‌ها باید کوتاه و روان باشد.
۳. در پایان، سلب مسئولیت اضافه کن."""

    print("\n==================================================")
    print("🧪 تست ۱: بررسی شخصیت و پرامپت سیستمی")
    print("سوال: سلام. لطفا خودت را معرفی کن.")
    print("--------------------------------------------------")
    ans1 = generate_response(model, tokenizer, sys_prompt, "سلام. لطفا خودت را معرفی کن.")
    print(ans1)

    print("\n==================================================")
    print("🧪 تست ۲: بررسی یادگیری سبک داده‌های ما (متن رای)")
    print("سوال: من یک پرونده در گروه حقوقی با شماره 140306390000086224 دارم. لطفا متن رای این پرونده را تحلیل کن.")
    print("--------------------------------------------------")
    ans2 = generate_response(model, tokenizer, sys_prompt, "من یک پرونده در گروه حقوقی با شماره 140306390000086224 دارم. لطفا متن رای این پرونده را تحلیل کن.")
    print(ans2)

    print("\n==================================================")
    print("🧪 تست ۳: بررسی دانش عمومی حقوقی (بدون سلب مسئولیت)")
    print("سوال: تفاوت اصلی بین عقد جایز و عقد لازم در قانون مدنی چیست؟")
    print("--------------------------------------------------")
    # اینجا پرامپت سیستمی را خالی می‌گذاریم تا ببینیم خود مدل چقدر می‌داند
    ans3 = generate_response(model, tokenizer, "", "تفاوت اصلی بین عقد جایز و عقد لازب در قانون مدنی چیست؟")
    print(ans3)
    print("==================================================\n")

if __name__ == "__main__":
    main()
