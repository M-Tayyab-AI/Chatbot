"""Lead capture: validation + a local fallback store (Neon store lives in services/db.py)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from placidway_bot.config import DATA_DIR

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
PHONE = re.compile(r"^[+\d][\d\s().-]{5,20}$")


def validate_lead(name: str, email: str, phone: str = "") -> list[str]:
    errors = []
    if len(name.strip()) < 2:
        errors.append("Please enter your name.")
    if not EMAIL.match(email.strip()):
        errors.append("Please enter a valid email address.")
    if phone.strip() and not PHONE.match(phone.strip()):
        errors.append("Please enter a valid phone number (or leave it empty).")
    return errors


class JsonlLeadStore:
    def __init__(self, path: Path = DATA_DIR / "logs" / "leads.jsonl"):
        self.path = path

    def save(self, **lead) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **lead},
                                ensure_ascii=False) + "\n")

    def recent(self, limit: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows[::-1][:limit]
