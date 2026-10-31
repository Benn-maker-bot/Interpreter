import json
import gzip
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
LOG_PATH = APP_DIR / "interpreter.log.jsonl"
ERROR_LOG_PATH = APP_DIR / "interpreter.error.log.jsonl"
SUMMARY_PATH = APP_DIR / "interpreter.log.summary.json"


def iter_log_files(base_path: Path):
    files = []
    if base_path.exists():
        files.append(base_path)
    files.extend(sorted(base_path.parent.glob(base_path.name + ".*.gz")))
    return files


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_summary():
    levels = Counter()
    events = Counter()
    issues = []
    total = 0
    for path in iter_log_files(LOG_PATH):
        for entry in read_jsonl(path):
            total += 1
            levels[entry.get("level", "UNKNOWN")] += 1
            events[entry.get("event", "unknown")] += 1
            if entry.get("level") in {"WARNING", "ERROR", "CRITICAL"}:
                issues.append({
                    "ts": entry.get("ts"),
                    "event": entry.get("event"),
                    "message": entry.get("message"),
                })
    summary = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "log_path": str(LOG_PATH),
        "error_log_path": str(ERROR_LOG_PATH),
        "total_entries": total,
        "levels": dict(levels),
        "top_events": dict(events.most_common(25)),
        "recent_issues": issues[-25:],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = build_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
