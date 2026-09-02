"""Text and prosody catalog for the Qingxiaojia v1 synthetic source stage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .hard_negatives import HardNegativeGenerator


POSITIVE_SPEEDS = (0.90, 0.95, 1.00, 1.05, 1.10)

NEGATIVE_PHRASES = (
    "今天天气不错",
    "请打开客厅的灯",
    "播放一首音乐",
    "现在几点了",
    "明天早上提醒我开会",
    "把空调温度调低一点",
    "帮我查一下今天的日程",
    "今天晚饭吃什么",
    "窗外好像开始下雨了",
    "请把电视声音调小",
    "我刚刚收到一条消息",
    "周末我们一起去公园吧",
    "这本书放在桌子上",
    "厨房里的水已经烧开了",
    "记得出门的时候带钥匙",
    "附近有没有好吃的面馆",
    "下一班地铁还有多久",
    "请导航到最近的停车场",
    "我想听今天的新闻",
    "会议改到下午三点",
    "你好",
    "青小甲",
    "您好",
    "你好吗",
    "你好，小明",
    "你好，小李",
    "你好，小王",
    "你好，小张",
    "早上好，小甲",
    "晚上好，小甲",
    "小甲，你好",
    "清早起来空气很好",
    "青年人喜欢新的挑战",
    "请小佳稍等一下",
    "家里的灯还没有关",
    "嘉宾已经到达现场",
    "加入购物清单",
    "轻轻地把门关上",
    "我没有听清你刚才的话",
    "请再说一遍",
)


@dataclass(frozen=True, slots=True)
class SourceUtteranceSpec:
    utterance_id: str
    label: str
    text: str
    synthesis_text: str
    speed: float
    hard_negative_tier: int | None = None


def with_terminal_boundary(text: str) -> str:
    return text if text.endswith(("。", "！", "？", "……", ".", "!", "?")) else f"{text}。"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def source_utterance_specs(wake_word: str) -> list[SourceUtteranceSpec]:
    specs: list[SourceUtteranceSpec] = []
    for index, speed in enumerate(POSITIVE_SPEEDS):
        specs.append(
            SourceUtteranceSpec(
                utterance_id=f"positive-{index:03d}",
                label="positive",
                text=wake_word,
                synthesis_text=with_terminal_boundary(wake_word),
                speed=speed,
            )
        )
    for index, phrase in enumerate(NEGATIVE_PHRASES):
        specs.append(
            SourceUtteranceSpec(
                utterance_id=f"negative-{index:03d}",
                label="negative",
                text=phrase,
                synthesis_text=with_terminal_boundary(phrase),
                speed=POSITIVE_SPEEDS[index % len(POSITIVE_SPEEDS)],
            )
        )
    for index, phrase in enumerate(HardNegativeGenerator().generate()):
        specs.append(
            SourceUtteranceSpec(
                utterance_id=f"hard-negative-{index:03d}-{_text_hash(phrase.text)}",
                label="hard_negative",
                text=phrase.text,
                synthesis_text=with_terminal_boundary(phrase.text),
                speed=POSITIVE_SPEEDS[index % len(POSITIVE_SPEEDS)],
                hard_negative_tier=phrase.tier,
            )
        )
    return specs
