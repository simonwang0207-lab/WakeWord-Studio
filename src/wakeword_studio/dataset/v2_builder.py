"""Speaker-disjoint, streaming-oriented qingxiaojia_v2 dataset builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import TARGET_PCM_SUBTYPE, TARGET_SAMPLE_RATE_HZ, load_audio_float32, resample_audio
from .formal_builder import _fit_audio, _procedural_noise, _stable_seed
from .manifest import AcousticMetadata, DatasetManifest, DatasetRecord, SpeakerMetadata, sha256_file


SPLITS = ("train", "validation", "test")
LABELS = ("positive", "negative", "hard_negative", "ambient")
SPEECH_FAMILIES = ("kokoro", "voxcpm15")


def _json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _trim_silence(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(samples):
        return np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(samples)))
    threshold = max(0.003, peak * 0.02)
    active = np.flatnonzero(np.abs(samples) >= threshold)
    if not len(active):
        return samples
    pad = int(0.04 * TARGET_SAMPLE_RATE_HZ)
    start = max(0, int(active[0]) - pad)
    end = min(len(samples), int(active[-1]) + pad + 1)
    return samples[start:end]


def _duration_bin(duration: float) -> str:
    if duration < 2.2:
        return "1.5-2.2"
    if duration < 3.0:
        return "2.2-3.0"
    if duration < 4.0:
        return "3.0-4.0"
    return "4.0-5.0"


class V2DatasetBuilder:
    def __init__(self, config: dict[str, object], source_manifests: list[Path]) -> None:
        self.config = config
        self.sources: list[dict[str, object]] = []
        self.source_manifest_hashes: dict[str, str] = {}
        for manifest_path in source_manifests:
            manifest_path = manifest_path.resolve()
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            root = Path(str(raw.get("root") or manifest_path.parent)).resolve()
            self.source_manifest_hashes[str(manifest_path)] = sha256_file(manifest_path)
            for record in raw["records"]:
                item = dict(record)
                item["absolute_path"] = str((root / str(record["path"])).resolve())
                self.sources.append(item)
        self._audio_cache: dict[str, np.ndarray] = {}
        self._speed_cache: dict[tuple[int, float], np.ndarray] = {}
        self._candidate_cache: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        self._validate_source_allocation()

    def _validate_source_allocation(self) -> None:
        expected = self.config["speaker_allocation"]
        actual: dict[str, dict[str, set[str]]] = {
            split: {family: set() for family in SPEECH_FAMILIES} for split in SPLITS
        }
        group_splits: dict[str, set[str]] = {}
        utterance_splits: dict[str, set[str]] = {}
        for row in self.sources:
            split = str(row["split"])
            family = str(row["source_family"])
            if split not in SPLITS or family not in SPEECH_FAMILIES:
                raise ValueError(f"Unexpected source allocation: {split}/{family}")
            actual[split][family].add(str(row["speaker_id"]))
            group_splits.setdefault(str(row["source_group_id"]), set()).add(split)
            utterance_splits.setdefault(str(row["source_utterance_id"]), set()).add(split)
        leaking_groups = {key: value for key, value in group_splits.items() if len(value) > 1}
        leaking_utterances = {
            key: value for key, value in utterance_splits.items() if len(value) > 1
        }
        if leaking_groups or leaking_utterances:
            raise ValueError(
                f"Source leakage before augmentation: groups={leaking_groups}, "
                f"utterances={leaking_utterances}"
            )
        for split in SPLITS:
            for family in SPEECH_FAMILIES:
                wanted = set(expected[split][family])
                if actual[split][family] != wanted:
                    raise ValueError(
                        f"{split}/{family} speakers={sorted(actual[split][family])}; "
                        f"expected={sorted(wanted)}"
                    )

    def _load_source(self, record: dict[str, object]) -> np.ndarray:
        path = str(record["absolute_path"])
        if path not in self._audio_cache:
            audio, rate = load_audio_float32(Path(path))
            if rate != TARGET_SAMPLE_RATE_HZ:
                raise RuntimeError(f"Canonical source load failed: {path}")
            self._audio_cache[path] = _trim_silence(audio)
        return self._audio_cache[path]

    def _candidates(self, split: str, label: str, family: str) -> list[dict[str, object]]:
        key = (split, label, family)
        if key not in self._candidate_cache:
            rows = [
                row
                for row in self.sources
                if row["split"] == split
                and row["label"] == label
                and row["source_family"] == family
            ]
            rows.sort(key=lambda row: (str(row["speaker_id"]), str(row["record_id"])))
            if not rows:
                raise ValueError(f"No source candidate for {split}/{label}/{family}")
            self._candidate_cache[key] = rows
        return self._candidate_cache[key]

    def _place(
        self, audio: np.ndarray, index: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, int, int, str]:
        speed = (0.92, 0.97, 1.0, 1.03, 1.08)[(index // 10) % 5]
        speed_key = (id(audio), speed)
        if speed_key not in self._speed_cache:
            speech = resample_audio(
                audio,
                TARGET_SAMPLE_RATE_HZ,
                max(1, int(TARGET_SAMPLE_RATE_HZ / speed)),
            )
            maximum_speech = int(4.8 * TARGET_SAMPLE_RATE_HZ)
            if len(speech) > maximum_speech:
                source_rate = max(
                    1, int(TARGET_SAMPLE_RATE_HZ * len(speech) / maximum_speech)
                )
                speech = resample_audio(speech, source_rate, TARGET_SAMPLE_RATE_HZ)
            self._speed_cache[speed_key] = np.asarray(speech, dtype=np.float32)
        speech = self._speed_cache[speed_key]
        slot = index % 10
        if slot < 3:
            placement = "front"
        elif slot < 7:
            placement = "middle"
        else:
            placement = "back"

        # Use the same explicit duration schedule in every label/split.  Padding
        # is added without truncating speech; unusually long source utterances
        # may move into a larger observed bin but never get clipped.
        duration_slot = (index * 7) % 20
        if duration_slot < 5:
            target_duration = float(rng.uniform(1.5, 2.2))
        elif duration_slot < 12:
            target_duration = float(rng.uniform(2.2, 3.0))
        elif duration_slot < 18:
            target_duration = float(rng.uniform(3.0, 4.0))
        else:
            target_duration = float(rng.uniform(4.0, 5.0))
        speech_seconds = len(speech) / TARGET_SAMPLE_RATE_HZ
        # Never truncate a phrase.  A source longer than its scheduled bin is
        # promoted to the smallest duration that contains it.
        target_duration = min(5.0, max(target_duration, speech_seconds + 0.08))
        padding = max(0.0, target_duration - speech_seconds)
        if placement == "front":
            leading = padding * float(rng.uniform(0.02, 0.18))
        elif placement == "middle":
            leading = padding * float(rng.uniform(0.35, 0.65))
        else:
            leading = padding * float(rng.uniform(0.82, 0.98))
        trailing = padding - leading

        leading_samples = int(round(leading * TARGET_SAMPLE_RATE_HZ))
        trailing_samples = int(round(trailing * TARGET_SAMPLE_RATE_HZ))
        placed = np.concatenate(
            (
                np.zeros(leading_samples, dtype=np.float32),
                np.asarray(speech, dtype=np.float32),
                np.zeros(trailing_samples, dtype=np.float32),
            )
        )
        return placed, leading_samples, leading_samples + len(speech), placement

    def _background_speech(self, split: str, index: int, length: int) -> tuple[np.ndarray, str]:
        candidates = [
            row for row in self.sources if row["split"] == split and row["label"] == "negative"
        ]
        candidates.sort(key=lambda row: str(row["record_id"]))
        source = candidates[index % len(candidates)]
        return _fit_audio(self._load_source(source), length), str(source["record_id"])

    def _mix(
        self,
        placed: np.ndarray,
        split: str,
        index: int,
        start: int,
        end: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, str, float | None, str | None]:
        gain_db = float(rng.uniform(-5.0, 3.0))
        signal = np.asarray(placed, dtype=np.float32) * 10 ** (gain_db / 20.0)
        condition = str(self.config["noise_schedule"][index % 5])
        if condition == "clean":
            return np.clip(signal, -1.0, 1.0), gain_db, "clean", None, None
        snr_db = float(condition)
        categories = list(self.config["noise_categories"])
        category = str(categories[(index // 5) % len(categories)])
        if category == "tv_speech":
            noise, source_id = self._background_speech(split, index, len(signal))
            noise_id = f"tv_speech:{source_id}"
        else:
            noise = _procedural_noise(category, len(signal), rng)
            noise_id = f"procedural_{category}:{split}:{index:06d}"
        active = signal[start:end] if end > start else signal
        signal_rms = float(np.sqrt(np.mean(np.square(active)) + 1e-12))
        noise_rms = float(np.sqrt(np.mean(np.square(noise)) + 1e-12))
        mixed = signal + noise * (signal_rms / (10 ** (snr_db / 20.0) * noise_rms))
        reverb_id: str | None = None
        if category in {"room", "office", "device"}:
            delay_seconds = {"room": 0.045, "office": 0.08, "device": 0.018}[category]
            delay = int(TARGET_SAMPLE_RATE_HZ * delay_seconds)
            if len(mixed) > delay:
                reflected = mixed.copy()
                reflected[delay:] += mixed[:-delay] * 0.18
                mixed = reflected
            reverb_id = f"synthetic_{category}_response"
        return np.clip(mixed, -1.0, 1.0), gain_db, noise_id, snr_db, reverb_id

    def _write(self, path: Path, audio: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        partial = path.with_name(f"{path.stem}.partial.wav")
        sf.write(partial, audio, TARGET_SAMPLE_RATE_HZ, subtype=TARGET_PCM_SUBTYPE)
        partial.replace(path)

    def build(self, output_root: Path, limit: int | None = None) -> DatasetManifest:
        output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        state = {
            "config_hash": _json_hash(self.config),
            "source_manifest_hashes": self.source_manifest_hashes,
            "builder_sha256": sha256_file(Path(__file__).resolve()),
        }
        state_path = output_root / "build_state.json"
        if state_path.exists():
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if previous != state:
                raise RuntimeError("Existing v2 build state differs from config/source manifests")

        expected_full = sum(
            int(count)
            for split_counts in self.config["counts"].values()
            for count in split_counts.values()
        )
        expected = min(expected_full, limit) if limit else expected_full
        completed = 0
        records: list[DatasetRecord] = []
        speech_stream_indices = {split: 0 for split in SPLITS}
        for split in SPLITS:
            for label in LABELS:
                count = int(self.config["counts"][split][label])
                for index in range(count):
                    if completed >= expected:
                        break
                    record_id = f"{split}-{label}-{index:06d}"
                    rng = np.random.default_rng(
                        _stable_seed(f"{self.config['seed']}:{record_id}")
                    )
                    path = output_root / split / label / f"{record_id}.wav"
                    if label == "ambient":
                        duration = float(rng.uniform(1.5, 5.0))
                        length = int(round(duration * TARGET_SAMPLE_RATE_HZ))
                        category = str(
                            self.config["noise_categories"][index % len(self.config["noise_categories"])]
                        )
                        audio = _procedural_noise(category, length, rng)
                        peak = float(np.max(np.abs(audio)) + 1e-12)
                        audio = np.asarray(audio / peak * 0.35, dtype=np.float32)
                        source = {
                            "text": None,
                            "speaker_id": "none",
                            "source_family": "procedural_ambient",
                            "source_group_id": f"ambient:{split}",
                            "record_id": f"ambient:{split}:{index:06d}",
                            "gender": None,
                            "age_group": None,
                            "age_source": "unknown",
                            "hard_negative_tier": None,
                            "speed": None,
                        }
                        acoustic = AcousticMetadata(
                            noise_id=f"procedural_{category}:{split}:{index:06d}",
                            phrase_placement="background_only",
                            duration_bin=_duration_bin(duration),
                            window_alignment="no_wake_phrase",
                        )
                    else:
                        family = SPEECH_FAMILIES[index % len(SPEECH_FAMILIES)]
                        candidates = self._candidates(split, label, family)
                        source = candidates[(index // len(SPEECH_FAMILIES)) % len(candidates)]
                        raw = self._load_source(source)
                        stream_index = speech_stream_indices[split]
                        speech_stream_indices[split] += 1
                        placed, start, end, placement = self._place(raw, stream_index, rng)
                        audio, gain_db, noise_id, snr_db, reverb_id = self._mix(
                            placed, split, stream_index, start, end, rng
                        )
                        duration = len(audio) / TARGET_SAMPLE_RATE_HZ
                        acoustic = AcousticMetadata(
                            speaking_rate=(
                                None if source.get("speed") is None else float(source["speed"])
                            ),
                            gain_db=gain_db,
                            noise_id=noise_id,
                            snr_db=snr_db,
                            reverb_id=reverb_id,
                            leading_silence_seconds=round(start / TARGET_SAMPLE_RATE_HZ, 6),
                            trailing_silence_seconds=round(
                                (len(audio) - end) / TARGET_SAMPLE_RATE_HZ, 6
                            ),
                            utterance_start_ms=round(start / TARGET_SAMPLE_RATE_HZ * 1000, 3),
                            utterance_end_ms=round(end / TARGET_SAMPLE_RATE_HZ * 1000, 3),
                            phrase_start_ms=(
                                round(start / TARGET_SAMPLE_RATE_HZ * 1000, 3)
                                if label == "positive"
                                else None
                            ),
                            phrase_end_ms=(
                                round(end / TARGET_SAMPLE_RATE_HZ * 1000, 3)
                                if label == "positive"
                                else None
                            ),
                            phrase_placement=placement,
                            duration_bin=_duration_bin(duration),
                            window_alignment=(
                                "complete_phrase_interval" if label == "positive" else "no_wake_phrase"
                            ),
                        )

                    self._write(path, audio)
                    info = sf.info(path)
                    speaker = SpeakerMetadata(
                        speaker_id=str(source["speaker_id"]),
                        source=str(source["source_family"]),
                        gender=source.get("gender"),
                        age_group=source.get("age_group"),
                        age_source=str(source.get("age_source", "unknown")),
                        reference_speaker_id=source.get("reference_speaker_id"),
                        reference_age_group=source.get("reference_age_group"),
                        reference_age_group_source=source.get("reference_age_group_source"),
                        perceived_age_verified=bool(source.get("perceived_age_verified", False)),
                    )
                    records.append(
                        DatasetRecord(
                            record_id=record_id,
                            audio_path=path.relative_to(output_root).as_posix(),
                            label=label,
                            split=split,
                            text=source.get("text"),
                            speaker=speaker,
                            acoustic=acoustic,
                            sample_rate_hz=int(info.samplerate),
                            original_sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
                            duration_seconds=round(float(info.duration), 6),
                            sha256=sha256_file(path),
                            source_utterance_id=str(source["record_id"]),
                            source_group_id=str(source["source_group_id"]),
                            augmentation_id=f"v2-placement-noise-{index:06d}",
                            hard_negative_tier=(
                                None
                                if source.get("hard_negative_tier") is None
                                else int(source["hard_negative_tier"])
                            ),
                        )
                    )
                    completed += 1
                    if completed % 50 == 0 or completed == expected:
                        print(
                            f"V2 BUILD HEARTBEAT records={completed}/{expected} "
                            f"split={split} label={label}",
                            flush=True,
                        )
                    if completed % 500 == 0:
                        partial = DatasetManifest(
                            wake_word=str(self.config["wake_word"]),
                            records=records,
                            source_kind="mixed",
                            root=str(output_root),
                            generator={"config": self.config, "state": state, "partial": True},
                        )
                        partial.save(output_root / "DatasetManifest.partial.json")
                if completed >= expected:
                    break
            if completed >= expected:
                break

        manifest = DatasetManifest(
            wake_word=str(self.config["wake_word"]),
            records=records,
            source_kind="mixed",
            root=str(output_root),
            generator={
                "config": self.config,
                "state": state,
                "streaming_window_labels": "derive from phrase_start_ms/phrase_end_ms",
            },
            coverage_policy={
                "required_labels": list(LABELS),
                "speaker_disjoint": True,
                "reference_age_rule": (
                    "AISHELL-3 A/B/C/D are demographic metadata only; perceptual age is unverified."
                ),
                "perceived_age_verified": False,
                "streaming_alignment": "positive phrase interval stored exactly after placement",
            },
        )
        manifest_name = "DatasetManifest.json" if expected == expected_full else "DatasetManifest.pilot.json"
        manifest.save(output_root / manifest_name)
        if expected == expected_full:
            state_partial = output_root / "build_state.partial.json"
            state_partial.write_text(json.dumps(state, indent=2), encoding="utf-8")
            state_partial.replace(state_path)
            (output_root / "DatasetManifest.partial.json").unlink(missing_ok=True)
        print(f"V2 BUILD COMPLETE records={len(records)} manifest={manifest_name}", flush=True)
        return manifest
