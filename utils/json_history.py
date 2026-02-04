import json
import os
from typing import List, Dict, Optional

BASE_DIR = "chat_history"
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
os.makedirs(BASE_DIR, exist_ok=True)


def _load_all() -> Dict[str, List[Dict]]:
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        return {}
    
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        # Handle corrupted file - optionally log warning
        return {}



def _save_all(data: Dict[str, List[Dict]]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load(thread_id: str, last_n: Optional[int] = None) -> List[Dict]:
    all_data = _load_all()
    data = all_data.get(thread_id, [])
    return data[-last_n:] if last_n else data


def append(thread_id: str, entries: List[Dict]) -> None:
    all_data = _load_all()
    if thread_id not in all_data:
        all_data[thread_id] = []
    all_data[thread_id].extend(entries)
    _save_all(all_data)


def clear(thread_id: str) -> None:
    all_data = _load_all()
    all_data[thread_id] = []
    _save_all(all_data)
