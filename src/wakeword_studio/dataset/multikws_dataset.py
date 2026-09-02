"""Configuration contract for vocabulary-driven Multi-KWS dataset builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


PROFILE_DEFAULTS: dict[str, dict[str, int]] = {
    "quick": {
        "positive_per_keyword": 4,
        "background_speech_count": 8,
        "hard_negative_count": 4,
        "ambient_count": 4,
    },
    "small": {
        "positive_per_keyword": 80,
        "background_speech_count": 240,
        "hard_negative_count": 120,
        "ambient_count": 120,
    },
}


@dataclass(frozen=True, slots=True)
class MultiKWSDatasetPlan:
    profile: str
    positive_per_keyword: int
    background_speech_count: int
    hard_negative_count: int
    ambient_count: int
    seed: int

    @property
    def background_count(self) -> int:
        return self.background_speech_count + self.hard_negative_count + self.ambient_count


def resolve_dataset_plan(config: Mapping[str, Any]) -> MultiKWSDatasetPlan:
    """Resolve quick/small/formal/custom without hard-coding a formal size."""

    profile = str(config.get("profile", "custom")).lower()
    if profile not in {"quick", "small", "formal", "custom"}:
        raise ValueError("profile must be quick, small, formal, or custom")
    defaults = PROFILE_DEFAULTS.get(profile, {})

    def count(name: str) -> int:
        value = config.get(name, defaults.get(name))
        if value is None:
            if profile == "formal":
                raise ValueError(
                    "formal counts must be chosen after auditing provider speed, disk, and training time"
                )
            raise ValueError(f"custom profile requires {name}")
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be positive")
        return value

    return MultiKWSDatasetPlan(
        profile=profile,
        positive_per_keyword=count("positive_per_keyword"),
        background_speech_count=count("background_speech_count"),
        hard_negative_count=count("hard_negative_count"),
        ambient_count=count("ambient_count"),
        seed=int(config.get("seed", 0)),
    )


def build_multikws_dataset(
    config: Mapping[str, Any],
    *,
    keyword_count: int,
    builder: Callable[[MultiKWSDatasetPlan, int], Any],
) -> Any:
    """Code-level entry point used by providers to materialize a resolved plan."""

    if keyword_count < 1:
        raise ValueError("keyword_count must be positive")
    return builder(resolve_dataset_plan(config), int(keyword_count))
