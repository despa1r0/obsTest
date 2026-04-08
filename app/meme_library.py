from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class MemeMatch:
    file_name: str
    file_path: Path
    score: int
    matched_tags: list[str]
    required_tags: list[str]
    effect: str
    priority: int


class MemeLibrary:
    def __init__(self, memes_dir: str | Path) -> None:
        self._memes_dir = Path(memes_dir).resolve()
        self._entries = self._load_entries()

    def has_entries(self) -> bool:
        return bool(self._entries)

    def find_best_match(self, tags: Iterable[str]) -> MemeMatch | None:
        normalized_tags = {tag.strip().lower() for tag in tags if tag and tag.strip()}
        if not normalized_tags:
            return None

        best_match: MemeMatch | None = None
        best_rank: tuple[int, int, int, int] | None = None

        for entry in self._entries:
            required_tags = entry["required_tags"]
            if any(tag not in normalized_tags for tag in required_tags):
                continue

            entry_tags = set(entry["tags"])
            matched_tags = sorted(normalized_tags & entry_tags)
            if not matched_tags and not required_tags:
                continue

            rank = (
                entry["priority"],
                len(required_tags),
                len(matched_tags),
                len(entry_tags),
            )
            match = MemeMatch(
                file_name=entry["file_name"],
                file_path=entry["file_path"],
                score=len(matched_tags),
                matched_tags=matched_tags,
                required_tags=required_tags,
                effect=entry["effect"],
                priority=entry["priority"],
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
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
            entries: list[dict] = []
            for entry in images:
                image_name = entry.get("file")
                tags = self._normalize_list(entry.get("tags", []))
                required_tags = self._normalize_list(entry.get("required_tags", []))
                if not image_name or not (tags or required_tags):
                    continue

                image_path = self._memes_dir / image_name
                if not image_path.exists():
                    continue

                entries.append(
                    {
                        "file_name": image_name,
                        "file_path": image_path,
                        "tags": tags,
                        "required_tags": required_tags,
                        "effect": str(entry.get("effect", "fade")).strip().lower() or "fade",
                        "priority": int(entry.get("priority", 0)),
                    }
                )
            return entries

        return []

    @staticmethod
    def _normalize_list(values: Iterable[str]) -> list[str]:
        return sorted(
            {
                str(value).strip().lower()
                for value in values
                if str(value).strip()
            }
        )
