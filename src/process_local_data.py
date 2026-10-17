import json
import os

INPUT_FILE = os.path.expanduser("~/Data/datasets/courts_ruling.json")
OUTPUT_DIR = "data/shards"
CHUNK_RECORDS = 500  # هر فایل 500 رکورد برای آموزش در گیت‌هاب

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Reading 156MB JSON file (Please wait) ---")
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

print(f"Total records found: {len(data)}")
print(f"--- Splitting into chunks of {CHUNK_RECORDS} records ---")

file_count = 1
current_records = []

def save_chunk(records, count):
    filename = os.path.join(OUTPUT_DIR, f"legal_chunk_{count}.jsonl")
    with open(filename, 'w', encoding='utf-8') as f:
        for item in records:
            # استخراج دیکشنری داخلی (چون کلید اصلی شماره پرونده است)
            inner_data = list(item.values())[0] if isinstance(item, dict) else item
            
            # جستجوی هوشمند برای پیدا کردن متن اصلی رای
            text_content = ""
            # اگر کلیدهای مشخصی وجود داشت:
            for key in ["متن_رای", "متن", "رای", "گردش_کار", "متن_دادنامه"]:
                if key in inner_data and isinstance(inner_data[key], str) and len(inner_data[key]) > 50:
                    text_content = inner_data[key]
                    break
            
            # اگر متن پیدا نشد، کل اطلاعات پرونده را به متن تبدیل می‌کنیم (تا داده از دست نرود)
            if not text_content:
                text_content = json.dumps(inner_data, ensure_ascii=False)

            # ساخت فرمت نهایی برای آموزش
            record = {"text": text_content}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"✅ Saved {filename} with {len(records)} records.")

for item in data:
    current_records.append(item)
    if len(current_records) >= CHUNK_RECORDS:
        save_chunk(current_records, file_count)
        file_count += 1
        current_records = []

if current_records:
    save_chunk(current_records, file_count)

print("🎉 All chunks created successfully!")
