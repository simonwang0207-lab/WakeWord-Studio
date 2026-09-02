"""Tiered hard-negative curriculum for the Qingxiaojia wake phrase."""

from __future__ import annotations

from dataclasses import dataclass

from pypinyin import Style, pinyin


@dataclass(frozen=True, slots=True)
class HardNegativePhrase:
    text: str
    tier: int
    reason: str


def pinyin_signature(text: str) -> tuple[str, ...]:
    """Return a tone-aware signature while ignoring punctuation."""

    return tuple(
        syllables[0]
        for syllables in pinyin(
            text,
            style=Style.TONE3,
            neutral_tone_with_five=True,
            errors="ignore",
        )
        if syllables
    )


class HardNegativeGenerator:
    """Generate reviewed, acoustically distinguishable near-miss phrases."""

    TARGET = "你好，青小甲"
    EXCLUDED_EXACT_HOMOPHONES = (
        "你好，倾小甲",
        "你好，清小甲",
        "你好，轻小甲",
    )
    _REVIEWED = (
        HardNegativePhrase("你好，青小佳", 1, "final-tone contrast jia1 versus jia3"),
        HardNegativePhrase("你好，请小甲", 1, "qing-tone contrast qing3 versus qing1"),
        HardNegativePhrase("你好，青小架", 1, "final-tone contrast jia4 versus jia3"),
        HardNegativePhrase("你好，青小杰", 1, "final-syllable contrast jie2 versus jia3"),
        HardNegativePhrase("你好，金小甲", 1, "middle-syllable contrast jin1 versus qing1"),
        HardNegativePhrase("你好，星小甲", 1, "middle-syllable contrast xing1 versus qing1"),
        HardNegativePhrase("你好，小甲", 2, "missing qing syllable"),
        HardNegativePhrase("青小甲", 2, "missing greeting"),
        HardNegativePhrase("你好，青甲", 2, "missing xiao syllable"),
        HardNegativePhrase("你好吗，青小甲", 2, "inserted syllable near greeting"),
        HardNegativePhrase("你好，小安", 3, "other wake-like name"),
        HardNegativePhrase("你好，小瑞", 3, "other wake-like name"),
    )

    def generate(self) -> list[HardNegativePhrase]:
        target_signature = pinyin_signature(self.TARGET)
        phrases = {item.text: item for item in self._REVIEWED}
        collisions = [
            item.text
            for item in phrases.values()
            if pinyin_signature(item.text) == target_signature
        ]
        if collisions:
            raise ValueError(f"Hard negatives collide with target pinyin: {collisions}")
        return sorted(phrases.values(), key=lambda item: (item.tier, item.text))
