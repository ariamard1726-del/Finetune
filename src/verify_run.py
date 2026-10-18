import json
import os
import sys

ADAPTER = "models/lora_adapter"
STATUS_FILE = os.path.join(ADAPTER, "last_run.json")

if not os.path.exists(STATUS_FILE):
    print("❌ last_run.json missing: training did not report a result.")
    sys.exit(1)

with open(STATUS_FILE, "r", encoding="utf-8") as f:
    r = json.load(f)
print(json.dumps(r, ensure_ascii=False, indent=2))

run_id = os.environ.get("GITHUB_RUN_ID")
if run_id and str(r.get("run_id")) != str(run_id):
    print("❌ last_run.json is from an older run (stale). Aborting.")
    sys.exit(1)

status = r.get("status")
if status in ("ok", "partial"):
    has_weights = os.path.exists(os.path.join(ADAPTER, "adapter_model.safetensors")) or \
        os.path.exists(os.path.join(ADAPTER, "adapter_model.bin"))
    if not has_weights:
        print("❌ Adapter weights are missing.")
        sys.exit(1)
    print(f"✅ Run verified (status={status}).")
elif status in ("no_data", "skipped_empty"):
    print(f"ℹ️ Nothing trained (status={status}).")
else:
    print(f"❌ Unknown status: {status}")
    sys.exit(1)
