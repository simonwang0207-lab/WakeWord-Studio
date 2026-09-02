"""Product-facing dataset sizes with explicit label and frozen split counts."""

from __future__ import annotations

from dataclasses import dataclass



def _split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Dependency-free exact apportionment for the desktop UI."""

    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


SCALE_PRESETS: dict[str, dict[str, int]] = {
    # Small enough to verify the provider, audio contract and manifest end-to-end.
    "快速测试": {"positive": 3, "hard_negative": 3, "negative": 3, "ambient": 3},
    # Useful for pipeline/model experiments, but deliberately not called formal data.
    "小规模实验": {"positive": 250, "hard_negative": 225, "negative": 400, "ambient": 125},
    # Matches the frozen v2 formal dataset design actually used by this project.
    "正式训练": {"positive": 3800, "hard_negative": 3420, "negative": 6080, "ambient": 1900},
}


@dataclass(frozen=True, slots=True)
class ProductDatasetPlan:
    mode: str
    targets: dict[str, int]
    split_targets: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.targets.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "targets": dict(self.targets),
            "split_targets": dict(self.split_targets),
            "total": self.total,
            "test_frozen_at_generation": True,
        }


def build_product_plan(
    mode: str,
    *,
    custom_total: int | None = None,
    custom_targets: dict[str, int] | None = None,
) -> ProductDatasetPlan:
    if mode == "自定义":
        required = {"positive", "hard_negative", "negative", "ambient"}
        if custom_targets is not None:
            if set(custom_targets) != required:
                raise ValueError(f"自定义类别必须包含：{sorted(required)}")
            targets = {key: int(value) for key, value in custom_targets.items()}
            if any(value < 0 for value in targets.values()) or sum(targets.values()) < 10:
                raise ValueError("自定义类别数量必须为非负数，且总量至少为 10 条")
        else:
            if custom_total is None or custom_total < 10:
                raise ValueError("自定义总量至少为 10 条")
            weights = SCALE_PRESETS["正式训练"]
            weight_total = sum(weights.values())
            targets = {
                name: int(custom_total * count / weight_total) for name, count in weights.items()
            }
            targets["negative"] += custom_total - sum(targets.values())
    else:
        try:
            targets = dict(SCALE_PRESETS[mode])
        except KeyError as exc:
            raise ValueError(f"未知生成规模：{mode}") from exc
    split_targets = {"train": 0, "validation": 0, "test": 0}
    ratios = (
        {"train": 15 / 19, "validation": 2 / 19, "test": 2 / 19}
        if mode == "正式训练"
        else {"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3}
        if mode == "快速测试"
        else {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    for count in targets.values():
        for split, value in _split_counts(count, ratios).items():
            split_targets[split] += value
    return ProductDatasetPlan(mode=mode, targets=targets, split_targets=split_targets)
