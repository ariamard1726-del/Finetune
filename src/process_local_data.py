import glob
import hashlib
import json
import os
import random
import re

INPUT_FILE = os.path.expanduser("~/Data/datasets/courts_ruling.json")
OUTPUT_DIR = "data/shards"
VAL_FILE = "data/val.jsonl"

CHUNK_RECORDS = 80       # تعداد نمونه در هر شارد
MAX_SHARDS = 500         # سقف شارد (بیشتر از قبل چون نمونه‌ها کوچک‌ترند)
VAL_RATIO = 0.02
VAL_MIN = 20
SEED = 42
MAX_FACTS_CHARS = 1200
MAX_RULING_CHARS = 1800
MIN_TEXT_CHARS = 300

SYSTEM_PROMPT = (
    "تو یک دستیار حقوقی فارسی‌زبان هستی. فقط بر اساس متن و مواد قانونی "
    "که در اختیارت قرار می‌گیرد پاسخ بده و اگر اطلاعات کافی نیست، صریح بگو."
)

_TRANS = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی", "ـ": ""})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_ANON = [
    (re.compile(r"\b09\d{9}\b"), "[شماره‌تماس]"),
    (re.compile(r"\b14\d{14,16}\b"), "[شماره‌پرونده]"),
    (re.compile(r"((?:کد|شماره)\s*ملی\s*:?\s*)\d{10}"), r"\1[حذف‌شده]"),
    (re.compile(r"\b\d{16}\b"), "[شماره‌کارت]"),
]

_LAW = re.compile(
    r"ماده\s*\d+(?:\s*مکرر)?(?:\s*(?:و|،)\s*\d+)*\s*(?:از\s*)?"
    r"(?:قانون|آیین[‌ ]?نامه)\s+[^\.\n،؛:()]{2,60}"
)
_LAW_CUT = re.compile(r"\s(?:مصوب|و\s*ماده|ماده)\b")


def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.translate(_TRANS).translate(_DIGITS)
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def anonymize(text):
    for pattern, repl in _ANON:
        text = pattern.sub(repl, text)
    return text


def cut_start(text, limit):
    if len(text) <= limit:
        return text
    head = text[:limit]
    idx = max(head.rfind("."), head.rfind("؛"), head.rfind("\n"))
    if idx > limit // 2:
        head = head[: idx + 1]
    return head.strip()


def cut_end(text, limit):
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    m = re.search(r"[\.؛\n]\s*", tail)
    if m and m.end() < len(tail) // 2:
        tail = tail[m.end():]
    return tail.strip()


def cited_laws(text, max_items=8):
    found = []
    for m in _LAW.finditer(text):
        s = re.sub(r"\s+", " ", m.group(0)).strip()
        s = _LAW_CUT.split(s)[0].strip()
        if s and s not in found:
            found.append(s)
        if len(found) >= max_items:
            break
    return found


def make_sample(user, assistant):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def extract_record(item):
    """استخراج فیلدها از ساختار واقعی دادگان.
    هر آیتم: {case_id: {number, link, عنوان, پیام, متن رای, simplified text, ...}}
    """
    if not isinstance(item, dict):
        return None
    # اولین مقدار dict را برمی‌داریم
    for v in item.values():
        if isinstance(v, dict):
            return v
    return None


def build_samples(record):
    """ساخت نمونه‌های آموزشی از یک رکورد کامل."""
    if not record:
        return []

    title = normalize(record.get("عنوان", ""))
    message = normalize(record.get("پیام", ""))
    ruling = normalize(record.get("متن رای", ""))
    simplified = normalize(record.get("simplified text", ""))
    group = normalize(record.get("گروه", ""))
    court = normalize(record.get("مرجع صدور", ""))

    ruling = anonymize(ruling)
    simplified = anonymize(simplified)
    message = anonymize(message)

    samples = []

    # --- نوع ۱: از شرح پرونده به متن رأی ---
    if message and ruling and len(ruling) >= MIN_TEXT_CHARS:
        head_parts = []
        if title:
            head_parts.append(f"عنوان: {title}")
        if group:
            head_parts.append(f"گروه: {group}")
        if court:
            head_parts.append(f"مرجع صدور: {court}")
        header = "\n".join(head_parts)
        facts = cut_start(message, MAX_FACTS_CHARS)
        user = f"{header}\n\nشرح پرونده:\n{facts}\n\nبر اساس این وقایع، استدلال و رأی دادگاه را بنویس."
        samples.append(make_sample(user, cut_end(ruling, MAX_RULING_CHARS)))

    # --- نوع ۲: خلاصه‌سازی رأی ---
    if ruling and simplified and len(ruling) >= MIN_TEXT_CHARS and len(simplified) >= 100:
        user = f"متن رأی:\n{cut_end(ruling, MAX_RULING_CHARS)}\n\nاین رأی را به زبان ساده خلاصه کن."
        samples.append(make_sample(user, simplified))

    # --- نوع ۳: استخراج مواد قانونی ---
    if ruling and len(ruling) >= MIN_TEXT_CHARS:
        laws = cited_laws(ruling)
        if laws:
            user = (
                f"متن رأی:\n{cut_end(ruling, MAX_RULING_CHARS)}\n\n"
                "مواد قانونی که در این رأی به آن‌ها استناد شده را فهرست کن."
            )
            samples.append(make_sample(user, "\n".join(f"- {x}" for x in laws)))

    return samples


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    print("--- Reading JSON file ---")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    print(f"total records: {len(data)}")

    seen, samples, skipped = set(), [], 0
    type_counts = {1: 0, 2: 0, 3: 0}

    for item in data:
        record = extract_record(item)
        if not record:
            skipped += 1
            continue
        batch = build_samples(record)
        if not batch:
            skipped += 1
            continue
        for idx, s in enumerate(batch, 1):
            key = hashlib.md5(json.dumps(s, ensure_ascii=False).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            samples.append(s)
            if idx <= 3:
                type_counts[idx] += 1

    print(f"skipped: {skipped} | total samples: {len(samples)}")
    print(f"نوع ۱ (شرح→رأی): {type_counts[1]}, نوع ۲ (خلاصه): {type_counts[2]}, نوع ۳ (قوانین): {type_counts[3]}")

    if not samples:
        print("❌ No samples produced. Check data structure.")
        return

    rng = random.Random(SEED)
    rng.shuffle(samples)

    n_val = min(max(VAL_MIN, int(len(samples) * VAL_RATIO)), len(samples) // 10)
    val, train = samples[:n_val], samples[n_val:]
    train = train[: MAX_SHARDS * CHUNK_RECORDS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(OUTPUT_DIR, "legal_chunk_*.jsonl")):
        os.remove(old)

    write_jsonl(VAL_FILE, val)
    print(f"✅ val: {len(val)}")

    count = 0
    for i in range(0, len(train), CHUNK_RECORDS):
        count += 1
        path = os.path.join(OUTPUT_DIR, f"legal_chunk_{count}.jsonl")
        write_jsonl(path, train[i : i + CHUNK_RECORDS])

    print(f"✅ train: {len(train)} | shards: {count}")

    print("--- Preview (first sample) ---")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
