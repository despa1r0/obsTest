from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class MemeMatch:
    file_name: str
    score: int
    matched_tags: list[str]


class MemeLibrary:
    def __init__(self, memes_dir: str | Path) -> None:
        self._memes_dir = Path(memes_dir).resolve()
        self._entries = self._load_entries()

    def has_entries(self) -> bool:
        return bool(self._entries)

    def find_best_match(self, tags: Iterable[str]) -> MemeMatch | None:
        normalized_tags = {
            tag.strip().lower()
            for tag in tags
            if tag and tag.strip()
        }
        if not normalized_tags:
            return None

        best_match: MemeMatch | None = None
        for entry in self._entries:
            entry_tags = {tag.lower() for tag in entry.get("tags", [])}
            matched_tags = sorted(normalized_tags & entry_tags)
            if not matched_tags:
                continue

            match = MemeMatch(
                file_name=entry["file"],
                score=len(matched_tags),
                matched_tags=matched_tags,
            )
            if best_match is None or match.score > best_match.score:
                best_match = match

        return best_match

    def _load_entries(self) -> list[dict]:
        for file_name in ("tags.json", "tags.example.json"):
            tags_path = self._memes_dir / file_name
            if not tags_path.exists():
                continue

            with tags_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            images = payload.get("images", [])
            return [entry for entry in images if entry.get("file") and entry.get("tags")]

        return []
