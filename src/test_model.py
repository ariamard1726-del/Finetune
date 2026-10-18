import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "alphaedge-ai/Qwen3-1.7B-fas-32768"
ADAPTER = "models/lora_adapter"
SYSTEM = (
    "تو یک دستیار حقوقی فارسی‌زبان هستی. فقط بر اساس متن و مواد قانونی "
    "که در اختیارت قرار می‌گیرد پاسخ بده و اگر اطلاعات کافی نیست، صریح بگو."
)

QUESTIONS = [
    "اگر قرارداد شفاهی باشد، آیا در دادگاه قابل پیگیری است؟",
    "تفاوت اصلی بین عقد جایز و عقد لازم در قانون مدنی چیست؟",
    "وقایع: خواهان مدعی است مبلغ 500 میلیون ریال به خوانده قرض داده و خوانده پس نداده است.\n\n"
    "بر اساس این وقایع، استدلال و رأی دادگاه را بنویس.",
]


def build_prompt(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n"


def stop_ids(tok):
    ids = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    t = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(t, int) and t >= 0 and t != tok.unk_token_id and t not in ids:
        ids.append(t)
    return ids


def generate(model, tok, user, max_new_tokens=120):
    prompt = build_prompt(
        tok, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    )
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=stop_ids(tok),
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    print(f"--- Loading base model: {BASE} ---", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cpu"
    )

    has_adapter = os.path.exists(os.path.join(ADAPTER, "adapter_model.safetensors")) or \
        os.path.exists(os.path.join(ADAPTER, "adapter_model.bin"))

    if has_adapter:
        model = PeftModel.from_pretrained(model, ADAPTER, local_files_only=True)
        print("✅ Adapter loaded.", flush=True)
    else:
        print("⚠️ No adapter found. Testing base model only.", flush=True)
    model.eval()

    for i, q in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 60)
        print(f"سؤال {i}:\n{q}")
        if has_adapter:
            with model.disable_adapter():
                print("\n--- مدل پایه ---")
                print(generate(model, tok, q), flush=True)
            print("\n--- مدل + adapter ---")
            print(generate(model, tok, q), flush=True)
        else:
            print("\n--- مدل پایه ---")
            print(generate(model, tok, q), flush=True)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
