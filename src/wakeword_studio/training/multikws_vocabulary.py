"""Configuration-driven vocabulary and add-keyword contract for Multi-KWS."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def normalize_keyword(text: str) -> str:
    return re.sub(r"[\s，,。.!！？?、]+", "", text).strip()


@dataclass(frozen=True, slots=True)
class KeywordClass:
    keyword_id: str
    display_name: str
    normalized_text: str
    class_index: int
    aliases: tuple[str, ...]
    dataset: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MultiKWSVocabulary:
    vocabulary_id: str
    version: int
    background: KeywordClass
    keywords: tuple[KeywordClass, ...]
    add_keyword_requires_retrain: bool = True

    @property
    def num_classes(self) -> int:
        return 1 + len(self.keywords)

    @property
    def class_names(self) -> tuple[str, ...]:
        return (self.background.keyword_id, *(item.keyword_id for item in self.keywords))

    @classmethod
    def load(cls, path: Path) -> "MultiKWSVocabulary":
        raw = json.loads(path.read_text(encoding="utf-8"))
        enabled = [item for item in raw["keywords"] if item.get("enabled", True)]
        background_raw = raw["background"]

        def parse(item: dict[str, Any]) -> KeywordClass:
            return KeywordClass(
                keyword_id=str(item["keyword_id"]), display_name=str(item["display_name"]),
                normalized_text=str(item["normalized_text"]), class_index=int(item["class_index"]),
                aliases=tuple(str(value) for value in item.get("aliases", ())),
                dataset=dict(item.get("dataset", {})),
            )

        vocabulary = cls(
            vocabulary_id=str(raw["vocabulary_id"]), version=int(raw["version"]),
            background=parse(background_raw), keywords=tuple(parse(item) for item in enabled),
            add_keyword_requires_retrain=bool(raw.get("add_keyword_requires_retrain", True)),
        )
        vocabulary.validate()
        return vocabulary

    def validate(self) -> None:
        if self.background.class_index != 0:
            raise ValueError("background class_index must be 0")
        expected = list(range(1, self.num_classes))
        actual = [item.class_index for item in self.keywords]
        if actual != expected:
            raise ValueError(f"Enabled keyword class indices must be contiguous: {actual}")
        ids = self.class_names
        if len(set(ids)) != len(ids):
            raise ValueError("keyword_id values must be unique")
        texts = [item.normalized_text for item in self.keywords]
        if any(normalize_keyword(item.display_name) != item.normalized_text for item in self.keywords):
            raise ValueError("normalized_text does not match display_name")
        if len(set(texts)) != len(texts):
            raise ValueError("normalized_text values must be unique")
        if not self.add_keyword_requires_retrain:
            raise ValueError("Softmax vocabulary expansion must require retraining")


def add_keyword(
    vocabulary_path: Path,
    *,
    keyword_id: str,
    display_name: str,
    destination: Path | None = None,
    aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return/write an expanded vocabulary plus an explicit retraining plan."""

    raw = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    current = MultiKWSVocabulary.load(vocabulary_path)
    normalized = normalize_keyword(display_name)
    if keyword_id in current.class_names or normalized in {item.normalized_text for item in current.keywords}:
        raise ValueError("Keyword id/text already exists")
    raw["version"] = int(raw["version"]) + 1
    raw["keywords"].append({
        "keyword_id": keyword_id, "display_name": display_name,
        "normalized_text": normalized, "class_index": current.num_classes,
        "aliases": list(aliases), "enabled": True, "dataset": {"language": "zh-CN"},
    })
    raw["add_keyword_requires_retrain"] = True
    plan = {
        "vocabulary": raw,
        "old_num_classes": current.num_classes,
        "new_num_classes": current.num_classes + 1,
        "classifier_head_action": f"expand_{current.num_classes}_to_{current.num_classes + 1}",
        "replay_existing_keywords": True,
        "validation_recalibration_required": True,
        "ADD_KEYWORD_REQUIRES_RETRAIN": True,
    }
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan

