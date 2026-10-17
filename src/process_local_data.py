import json
import os

INPUT_FILE = os.path.expanduser("~/Data/datasets/courts_ruling.json")
OUTPUT_DIR = "data/shards"
CHUNK_RECORDS = 500 

os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = """[نقش و شخصیت]
تو یک «دستیار حقوقی ارشد، دقیق و فارسی‌زبان» هستی. وظیفه تو تحلیل مسائل حقوقی، استخراج قوانین مرتبط، استدلال منطقی و ارائه راهکارهای عملی است.
[قوانین طلایی]
۱. همیشه ساختار استدلال گام‌به‌گام داشته باش.
۲. پاسخ‌ها باید کوتاه و روان باشد.
۳. در پایان، سلب مسئولیت اضافه کن."""

print("--- Reading 156MB JSON file ---")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total records: {len(data)}")
file_count = 1
current_records = []

def save_chunk(records, count):
    filename = os.path.join(OUTPUT_DIR, f"legal_chunk_{count}.jsonl")
    with open(filename, 'w', encoding='utf-8') as f:
        for item in records:
            inner_data = list(item.values())[0] if isinstance(item, dict) else item
            
            text_content = ""
            for key in ["متن_رای", "متن", "رای", "گردش_کار", "متن_دادنامه"]:
                if key in inner_data and isinstance(inner_data[key], str) and len(inner_data[key]) > 50:
                    text_content = inner_data[key]
                    break
            if not text_content:
                text_content = json.dumps(inner_data, ensure_ascii=False)

            case_num = inner_data.get('شماره پرونده', 'نامشخص')
            group = inner_data.get('گروه', 'حقوقی')
            
            user_prompt = f"من یک پرونده در گروه {group} با شماره {case_num} دارم. لطفا متن رای این پرونده را تحلیل کن و خلاصه‌ای از استدلال دادگاه ارائه بده."
            
            formatted_text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{text_content}<|im_end|>"
            
            f.write(json.dumps({"text": formatted_text}, ensure_ascii=False) + '\n')
            
    print(f"✅ Saved {filename} with {len(records)} records.")

for item in data:
    current_records.append(item)
    if len(current_records) >= CHUNK_RECORDS:
        save_chunk(current_records, file_count)
        file_count += 1
        current_records = []

if current_records:
    save_chunk(current_records, file_count)

print("🎉 All Instruction-Tuned chunks created!")
