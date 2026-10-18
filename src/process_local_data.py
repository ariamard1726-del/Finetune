import glob
import hashlib
import json
import os
import random
import re

INPUT_FILE = os.path.expanduser("~/Data/datasets/courts_ruling.json")
OUTPUT_DIR = "data/shards"
VAL_FILE = "data/val.jsonl"

CHUNK_RECORDS = 80      # تعداد نمونه آموزشی در هر شارد
MAX_SHARDS = 30         # سقف تعداد شارد (جلوگیری از سنگین شدن ریپو)
VAL_RATIO = 0.02
VAL_MIN = 20
SEED = 42
MAX_FACTS_CHARS = 1200
MAX_RULING_CHARS = 1500
MIN_TEXT_CHARS = 300

SYSTEM_PROMPT = (
    "تو یک دستیار حقوقی فارسی‌زبان هستی. فقط بر اساس متن و مواد قانونی "
    "که در اختیارت قرار می‌گیرد پاسخ بده و اگر اطلاعات کافی نیست، صریح بگو."
)

TEXT_KEYS = ["متن_رای", "متن", "رای", "گردش_کار", "متن_دادنامه"]

_TRANS = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی", "ـ": ""})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_ANON = [
    (re.compile(r"\b09\d{9}\b"), "[شماره‌تماس]"),
    (re.compile(r"\b14\d{14,16}\b"), "[شماره‌پرونده]"),
    (re.compile(r"((?:کد|شماره)\s*ملی\s*:?\s*)\d{10}"), r"\1[حذف‌شده]"),
    (re.compile(r"\b\d{16}\b"), "[شماره‌کارت]"),
]

_RULING_MARK = re.compile(r"ر[اأ]ی\s+(?:دادگاه|شعبه|قاضی|دادرس)")
_LAW = re.compile(
    r"ماده\s*\d+(?:\s*مکرر)?(?:\s*(?:و|،)\s*\d+)*\s*(?:از\s*)?"
    r"(?:قانون|آیین[‌ ]?نامه)\s+[^\.\n،؛:()]{2,60}"
)
_LAW_CUT = re.compile(r"\s(?:مصوب|و\s*ماده|ماده)\b")


def normalize(text):
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


def build_samples(raw_text):
    text = anonymize(normalize(raw_text))
    samples = []

    mark = None
    for m in _RULING_MARK.finditer(text):
        mark = m

    if mark and mark.start() >= 200 and len(text) - mark.start() >= 150:
        facts = cut_start(text[: mark.start()].strip(), MAX_FACTS_CHARS)
        ruling = cut_end(text[mark.start():].strip(), MAX_RULING_CHARS)
        samples.append(make_sample(
            f"وقایع و گردش کار پرونده:\n{facts}\n\n"
            "بر اساس این وقایع، استدلال و رأی دادگاه را بنویس.",
            ruling,
        ))
    else:
        ruling = cut_end(text, MAX_RULING_CHARS)

    laws = cited_laws(ruling)
    if laws:
        samples.append(make_sample(
            f"متن رأی:\n{ruling}\n\n"
            "مواد قانونی که در این رأی به آن‌ها استناد شده را فهرست کن.",
            "\n".join(f"- {x}" for x in laws),
        ))
    return samples


def get_inner(item):
    if isinstance(item, dict):
        if any(k in item for k in TEXT_KEYS):
            return item
        for v in item.values():
            if isinstance(v, dict):
                return v
    return None


def get_text(inner):
    for key in TEXT_KEYS:
        v = inner.get(key)
        if isinstance(v, str) and len(v) >= MIN_TEXT_CHARS:
            return v
    return ""


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    print("--- Reading JSON file ---")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if len(data) == 1 and isinstance(next(iter(data.values())), list):
            data = next(iter(data.values()))
        else:
            data = list(data.values())

    seen, samples, skipped = set(), [], 0
    for item in data:
        inner = get_inner(item)
        text = get_text(inner) if inner else ""
        if not text:
            skipped += 1
            continue
        for s in build_samples(text):
            key = hashlib.md5(json.dumps(s, ensure_ascii=False).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            samples.append(s)

    print(f"records: {len(data)} | skipped: {skipped} | samples: {len(samples)}")
    if not samples:
        first = data[0] if data else None
        keys = list(first.keys())[:10] if isinstance(first, dict) else type(first)
        print("❌ No samples. First item keys:", keys)
        return

    random.Random(SEED).shuffle(samples)
    n_val = min(max(VAL_MIN, int(len(samples) * VAL_RATIO)), len(samples) // 10)
    val, train = samples[:n_val], samples[n_val:]
    train = train[: MAX_SHARDS * CHUNK_RECORDS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(OUTPUT_DIR, "legal_chunk_*.jsonl")):
        os.remove(old)

    write_jsonl(VAL_FILE, val)

    count = 0
    for i in range(0, len(train), CHUNK_RECORDS):
        count += 1
        path = os.path.join(OUTPUT_DIR, f"legal_chunk_{count}.jsonl")
        write_jsonl(path, train[i:i + CHUNK_RECORDS])

    print(f"✅ val: {len(val)} | train: {len(train)} | shards: {count}")
    print("--- Preview ---")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
