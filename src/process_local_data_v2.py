"""v2: بهبود برش + نرمال‌سازی ایین + حذف \xad"""
import glob, hashlib, json, os, random, re

INPUT_FILE = os.path.expanduser("~/Data/datasets/courts_ruling.json")
OUTPUT_DIR = "data/shards_v2"
VAL_FILE = "data/val_v2.jsonl"

CHUNK_RECORDS = 80
MAX_SHARDS = 500
VAL_RATIO = 0.02
VAL_MIN = 30
SEED = 42
MAX_FACTS_CHARS = 1200
MAX_RULING_CHARS = 1800
MIN_ASSISTANT_CHARS = 100
MIN_USER_CHARS = 50
MIN_TEXT_CHARS = 300

SYSTEM_PROMPT = (
    "تو یک دستیار حقوقی فارسی‌زبان هستی. فقط بر اساس متن و مواد قانونی "
    "که در اختیارت قرار می‌گیرد پاسخ بده و اگر اطلاعات کافی نیست، صریح بگو."
)

def deep_normalize(text):
    if not isinstance(text, str): return ""
    text = text.translate(str.maketrans({
        'ي': 'ی', 'ك': 'ک', 'ى': 'ی', 'ة': 'ه',
        'أ': 'ا', 'إ': 'ا', 'ؤ': 'و',
        '٠': '۰', '١': '۱', '٢': '۲', '٣': '۳', '٤': '۴',
        '٥': '۵', '٦': '۶', '٧': '۷', '٨': '۸', '٩': '۹',
    }))
    text = text.translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))
    # حذف کنترل + soft hyphen + bidi
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f\xad\u200e\u200f\u202a-\u202e\u2066-\u2069]', '', text)
    # اصلاح تایپو
    text = re.sub(r'صاد\s+ر', 'صادر', text)
    text = text.replace('ایین', 'آیین')
    # فاصله‌ها
    text = re.sub(r'[ \t\u00a0]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def anonymize(text):
    text = re.sub(r'\b09\d{9}\b', '[شماره‌تماس]', text)
    text = re.sub(r'\b14\d{14,16}\b', '[شماره‌پرونده]', text)
    text = re.sub(r'((?:کد|شماره)\s*ملی\s*:?\s*)\d{10}', r'\1[حذف‌شده]', text)
    text = re.sub(r'\b\d{16}\b', '[شماره‌کارت]', text)
    return text

def smart_cut_start(text, limit):
    """برش از ابتدا - حفظ مرز جمله"""
    if len(text) <= limit: return text
    head = text[:limit]
    idx = max(head.rfind('. '), head.rfind('! '), head.rfind('؟ '), head.rfind('\n'))
    if idx > limit * 0.5:
        head = head[:idx + 1]
    return head.strip()

def smart_cut_end(text, limit):
    """برش از انتها - حفظ مرز جمله"""
    if len(text) <= limit: return text
    tail = text[-limit:]

    # ۱. برش از ابتدا: اولین مرز جمله را پیدا کن و از آن شروع کن
    m = re.search(r'[.!؟]\s+', tail)
    if m and m.end() < len(tail) * 0.5:
        tail = tail[m.end():]

    # ۲. برش از انتها: آخرین مرز جمله را پیدا کن
    last_m = None
    for mm in re.finditer(r'[.!؟](?=\s|$)', tail):
        last_m = mm
    if last_m and last_m.end() > len(tail) * 0.3:
        tail = tail[:last_m.end()]
    else:
        # بدون نقطه: در مرز کلمه ببر
        sp = tail.rfind(' ')
        if sp > len(tail) * 0.7:
            tail = tail[:sp]

    return tail.strip()

def fix_bad_start(text):
    """حذف حروف ربط/علائم از ابتدا"""
    text = text.strip()
    # حذف مکرر و، ویرگول، نقطه
    while text and text[0] in 'و،؛:)]}.؟':
        text = text[1:].strip()
    return text

def make_sample(user, assistant):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }

_LAW = re.compile(
    r"ماده\s*\d+(?:\s*مکرر)?(?:\s*(?:و|،)\s*\d+)*\s*(?:از\s*)?"
    r"(?:قانون|آیین[‌ ]?نامه)\s+[^\.\n،؛:()]{2,60}"
)
_LAW_CUT = re.compile(r"\s(?:مصوب|و\s*ماده|ماده)\b")

def cited_laws(text, max_items=8):
    found = []
    for m in _LAW.finditer(text):
        s = re.sub(r'\s+', ' ', m.group(0)).strip()
        s = _LAW_CUT.split(s)[0].strip()
        if s and s not in found:
            found.append(s)
        if len(found) >= max_items: break
    return found

def extract_record(item):
    if not isinstance(item, dict): return None
    for v in item.values():
        if isinstance(v, dict): return v
    return None

def build_samples(record):
    if not record: return []
    title = deep_normalize(record.get("عنوان", ""))
    message = deep_normalize(record.get("پیام", ""))
    ruling = deep_normalize(record.get("متن رای", ""))
    simplified = deep_normalize(record.get("simplified text", ""))
    group = deep_normalize(record.get("گروه", ""))
    court = deep_normalize(record.get("مرجع صدور", ""))

    ruling = anonymize(ruling)
    simplified = anonymize(simplified)
    message = anonymize(message)

    samples = []

    # نوع ۱
    if message and ruling and len(ruling) >= MIN_TEXT_CHARS:
        head_parts = []
        if title: head_parts.append(f"عنوان: {title}")
        if group: head_parts.append(f"گروه: {group}")
        if court: head_parts.append(f"مرجع صدور: {court}")
        header = "\n".join(head_parts)
        facts = smart_cut_start(message, MAX_FACTS_CHARS)
        user = f"{header}\n\nشرح پرونده:\n{facts}\n\nبر اساس این وقایع، استدلال و رأی دادگاه را بنویس."
        asst = fix_bad_start(smart_cut_end(ruling, MAX_RULING_CHARS))
        if asst: samples.append(make_sample(user, asst))

    # نوع ۲
    if ruling and simplified and len(ruling) >= MIN_TEXT_CHARS and len(simplified) >= 100:
        user = f"متن رأی:\n{smart_cut_end(ruling, MAX_RULING_CHARS)}\n\nاین رأی را به زبان ساده خلاصه کن."
        asst = fix_bad_start(simplified)
        if asst: samples.append(make_sample(user, asst))

    # نوع ۳
    if ruling and len(ruling) >= MIN_TEXT_CHARS:
        laws = cited_laws(ruling)
        if laws:
            user = f"متن رأی:\n{smart_cut_end(ruling, MAX_RULING_CHARS)}\n\nمواد قانونی که در این رأی به آن‌ها استناد شده را فهرست کن."
            samples.append(make_sample(user, "\n".join(f"- {x}" for x in laws)))

    return samples

def is_high_quality(sample):
    u = sample['messages'][1]['content'].strip()
    a = sample['messages'][2]['content'].strip()
    if len(a) < MIN_ASSISTANT_CHARS: return False
    if len(u) < MIN_USER_CHARS: return False
    if a[0] in 'و،؛:)]}؟.': return False
    persian = len(re.findall(r'[\u0600-\u06FF]', a))
    if persian / len(a) < 0.5: return False
    if '؟؟' in a or '!!!' in a: return False
    return True

def dedup(samples):
    seen_exact = set(); seen_norm = set(); out = []
    for s in samples:
        a = s['messages'][2]['content']
        h1 = hashlib.md5(a.encode()).hexdigest()
        if h1 in seen_exact: continue
        norm = re.sub(r'\s+', ' ', a).lower()
        h2 = hashlib.md5(norm.encode()).hexdigest()
        if h2 in seen_norm: continue
        seen_exact.add(h1); seen_norm.add(h2); out.append(s)
    return out

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    print("--- Reading ---")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict): data = list(data.values())
    print(f"records: {len(data)}")

    all_samples = []; skipped = 0
    for item in data:
        rec = extract_record(item)
        if not rec: skipped += 1; continue
        batch = build_samples(rec)
        if not batch: skipped += 1; continue
        all_samples.extend(batch)

    print(f"raw: {len(all_samples)}, skipped records: {skipped}")
    filtered = [s for s in all_samples if is_high_quality(s)]
    print(f"after filter: {len(filtered)} (dropped {len(all_samples)-len(filtered)})")
    deduped = dedup(filtered)
    print(f"after dedup: {len(deduped)} (dropped {len(filtered)-len(deduped)})")

    if not deduped: print("❌ empty"); return

    rng = random.Random(SEED); rng.shuffle(deduped)
    n_val = min(max(VAL_MIN, int(len(deduped) * VAL_RATIO)), len(deduped) // 10)
    val, train = deduped[:n_val], deduped[n_val:]
    train = train[:MAX_SHARDS * CHUNK_RECORDS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(OUTPUT_DIR, "legal_chunk_*.jsonl")):
        os.remove(old)

    write_jsonl(VAL_FILE, val)
    print(f"✅ val: {len(val)}")

    count = 0
    for i in range(0, len(train), CHUNK_RECORDS):
        count += 1
        write_jsonl(os.path.join(OUTPUT_DIR, f"legal_chunk_{count}.jsonl"), train[i:i+CHUNK_RECORDS])
    print(f"✅ train: {len(train)} | shards: {count}")

if __name__ == "__main__":
    main()
