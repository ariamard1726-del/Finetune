import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "alphaedge-ai/Qwen3-1.7B-fas-32768"
ADAPTER = "models/lora_adapter"
VAL_FILE = "data/val.jsonl"
VAL_SAMPLES = 10
MAX_LENGTH = 1024
SYSTEM = (
    "تو یک دستیار حقوقی فارسی‌زبان هستی. فقط بر اساس متن و مواد قانونی "
    "که در اختیارت قرار می‌گیرد پاسخ بده و اگر اطلاعات کافی نیست، صریح بگو."
)


def build_prompt(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n"


def get_end_token(tok):
    t = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(t, int) and t >= 0 and t != tok.unk_token_id:
        return "<|im_end|>"
    return tok.eos_token or ""


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


def load_val(tok):
    items = []
    if not os.path.exists(VAL_FILE):
        return items
    end_token = get_end_token(tok)
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            p = tok(build_prompt(tok, msgs[:-1]), add_special_tokens=False)["input_ids"]
            a = tok(msgs[-1]["content"] + end_token, add_special_tokens=False)["input_ids"]
            if len(a) < 2 or len(p) + len(a) > MAX_LENGTH:
                continue
            items.append((p + a, [-100] * len(p) + a))
            if len(items) >= VAL_SAMPLES:
                break
    return items


def mean_loss(model, items):
    losses = []
    with torch.no_grad():
        for ids, labels in items:
            x = torch.tensor([ids])
            y = torch.tensor([labels])
            out = model(input_ids=x, attention_mask=torch.ones_like(x), labels=y)
            losses.append(out.loss.item())
    return sum(losses) / len(losses)


def main():
    has_adapter = os.path.exists(os.path.join(ADAPTER, "adapter_model.safetensors")) or \
        os.path.exists(os.path.join(ADAPTER, "adapter_model.bin"))
    if not has_adapter:
        print("❌ No adapter found. Run the training workflow first.")
        sys.exit(1)

    print("--- Loading model ---", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, ADAPTER, local_files_only=True)
    model.eval()

    exit_code = 0

    print("\n=== تست ۱: loss روی داده اعتبارسنجی ===", flush=True)
    items = load_val(tok)
    if not items:
        print("⚠️ val file missing or empty; skipping loss comparison.")
    else:
        with model.disable_adapter():
            base_loss = mean_loss(model, items)
        adapter_loss = mean_loss(model, items)
        print(f"samples: {len(items)}")
        print(f"base loss   : {base_loss:.4f}")
        print(f"adapter loss: {adapter_loss:.4f}")
        if adapter_loss <= base_loss:
            print("✅ PASS: adapter is better than or equal to the base model.")
        else:
            print("❌ FAIL: adapter is worse than the base model.")
            exit_code = 1

    print("\n=== تست ۲: استخراج مواد قانونی از متن رأی ===", flush=True)
    ruling = (
        "دادگاه با توجه به محتویات پرونده و به استناد ماده 10 قانون مدنی و "
        "ماده 219 قانون مدنی، دعوی خواهان را وارد دانسته و حکم به محکومیت خوانده صادر می‌نماید."
    )
    q2 = f"متن رأی:\n{ruling}\n\nمواد قانونی که در این رأی به آن‌ها استناد شده را فهرست کن."
    a2 = generate(model, tok, q2)
    print(a2)
    if "10" in a2 and "219" in a2:
        print("✅ both articles found")
    else:
        print("⚠️ expected articles 10 and 219 not both found (soft check)")

    print("\n=== تست ۳: ساختن رأی جعلی از روی شماره پرونده ===", flush=True)
    q3 = "رأی پرونده شماره 9999999999999999 چه بود؟"
    a3 = generate(model, tok, q3)
    print(a3)
    if "محکوم" in a3 or "حکم به" in a3:
        print("⚠️ possible fabricated ruling (soft check)")
    else:
        print("✅ no obvious fabricated ruling")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
