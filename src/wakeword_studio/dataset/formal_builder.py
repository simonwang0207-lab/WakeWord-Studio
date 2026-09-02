"""Resumable formal dataset assembly from canonical, speaker-exclusive TTS sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import (
    TARGET_PCM_SUBTYPE,
    TARGET_SAMPLE_RATE_HZ,
    load_audio_float32,
    resample_audio,
)
from .manifest import (
    AcousticMetadata,
    DatasetManifest,
    DatasetRecord,
    SpeakerMetadata,
    sha256_file,
)
from .planning import split_counts


def _json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _structural_config(config: dict[str, object]) -> dict[str, object]:
    """Exclude estimates that do not affect generated records or audio."""

    return {key: value for key, value in config.items() if key != "runtime_estimate"}


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def _fit_audio(audio: np.ndarray, length: int) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(samples):
        return np.zeros(length, dtype=np.float32)
    repeats = (length + len(samples) - 1) // len(samples)
    return np.tile(samples, repeats)[:length]


def _procedural_noise(category: str, length: int, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(length, dtype=np.float32) / TARGET_SAMPLE_RATE_HZ
    white = rng.normal(0.0, 1.0, length).astype(np.float32)
    if category == "room":
        return np.convolve(white, np.ones(24, dtype=np.float32) / 24, mode="same")
    if category == "fan":
        return 0.65 * np.sin(2 * np.pi * 55 * time) + 0.2 * white
    if category == "keyboard":
        noise = 0.03 * white
        for position in rng.integers(0, max(1, length - 160), size=max(1, length // 3200)):
            width = min(160, length - int(position))
            noise[int(position) : int(position) + width] += np.exp(
                -np.arange(width, dtype=np.float32) / 22.0
            ) * rng.choice((-1.0, 1.0))
        return noise
    if category == "office":
        return 0.55 * _procedural_noise("fan", length, rng) + 0.45 * _procedural_noise(
            "keyboard", length, rng
        )
    if category == "street":
        rumble = np.convolve(white, np.ones(80, dtype=np.float32) / 80, mode="same")
        return rumble + 0.15 * np.sin(2 * np.pi * 110 * time)
    if category == "music":
        phases = rng.uniform(0.0, 2 * np.pi, 3)
        return sum(
            np.sin(2 * np.pi * frequency * time + phase)
            for frequency, phase in zip((220.0, 277.18, 329.63), phases)
        ).astype(np.float32)
    return white


class FormalDatasetBuilder:
    def __init__(self, config: dict[str, object], source_manifests: list[Path]) -> None:
        self.config = config
        self.sources: list[dict[str, object]] = []
        self.source_manifest_hashes: dict[str, str] = {}
        for manifest_path in source_manifests:
            manifest_path = manifest_path.resolve()
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            root = Path(raw.get("root") or manifest_path.parent).resolve()
            self.source_manifest_hashes[str(manifest_path)] = sha256_file(manifest_path)
            for record in raw["records"]:
                item = dict(record)
                item["absolute_path"] = str((root / str(record["path"])).resolve())
                self.sources.append(item)
        self._audio_cache: dict[str, np.ndarray] = {}

    def _load_source(self, record: dict[str, object]) -> np.ndarray:
        path = str(record["absolute_path"])
        if path not in self._audio_cache:
            audio, sample_rate = load_audio_float32(Path(path))
            if sample_rate != TARGET_SAMPLE_RATE_HZ:
                raise RuntimeError(f"Source failed canonical load: {path}")
            self._audio_cache[path] = audio
        return self._audio_cache[path]

    def _background_speech(self, split: str, index: int, length: int) -> tuple[np.ndarray, str]:
        candidates = [
            row for row in self.sources if row["split"] == split and row["label"] == "negative"
        ]
        if not candidates:
            raise ValueError(f"No negative speech source available for split {split}")
        source = candidates[index % len(candidates)]
        return _fit_audio(self._load_source(source), length), str(source["record_id"])

    def _augment(
        self,
        audio: np.ndarray,
        split: str,
        label: str,
        index: int,
        record_id: str,
    ) -> tuple[np.ndarray, AcousticMetadata]:
        rng = np.random.default_rng(_stable_seed(f"{self.config['seed']}:{record_id}"))
        age_proxy: str | None = None
        proxy_bucket = index % 10
        if label != "ambient" and proxy_bucket == 7:
            audio = resample_audio(audio, TARGET_SAMPLE_RATE_HZ, int(TARGET_SAMPLE_RATE_HZ / 1.08))
            age_proxy = "higher_pitch_and_faster"
        elif label != "ambient" and proxy_bucket == 8:
            audio = resample_audio(audio, TARGET_SAMPLE_RATE_HZ, int(TARGET_SAMPLE_RATE_HZ / 0.93))
            age_proxy = "lower_pitch_and_slower"

        gain_db = float(rng.uniform(-5.0, 3.0))
        gained = np.asarray(audio, dtype=np.float32) * 10 ** (gain_db / 20.0)
        categories = list(self.config["noise_categories"])
        category = str(categories[index % len(categories)])
        if category == "clean":
            return np.clip(gained, -1.0, 1.0), AcousticMetadata(
                speaking_rate=None,
                gain_db=gain_db,
                noise_id="clean",
                acoustic_age_proxy=age_proxy,
            )

        if category == "tv_speech":
            noise, background_id = self._background_speech(split, index, len(gained))
            noise_id = f"tv_speech:{background_id}"
        else:
            noise = _procedural_noise(category, len(gained), rng)
            noise_id = f"procedural_{category}:{split}:{index:06d}"
        signal_rms = float(np.sqrt(np.mean(np.square(gained)) + 1e-12))
        noise_rms = float(np.sqrt(np.mean(np.square(noise)) + 1e-12))
        snr_values = list(self.config["snr_db"])
        snr_db = float(snr_values[(index // len(categories)) % len(snr_values)])
        mixed = gained + noise * (signal_rms / (10 ** (snr_db / 20.0) * noise_rms))
        reverb_id: str | None = None
        if category in {"room", "office"}:
            delay = int(TARGET_SAMPLE_RATE_HZ * (0.045 if category == "room" else 0.08))
            reverbed = mixed.copy()
            if len(mixed) > delay:
                reverbed[delay:] += mixed[:-delay] * 0.18
            mixed = reverbed
            reverb_id = f"synthetic_{category}_early_reflection"
        return np.clip(mixed, -1.0, 1.0), AcousticMetadata(
            speaking_rate=None,
            gain_db=gain_db,
            noise_id=noise_id,
            snr_db=snr_db,
            reverb_id=reverb_id,
            acoustic_age_proxy=age_proxy,
        )

    def _write(self, path: Path, audio: np.ndarray, *, overwrite: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            return
        partial = path.with_name(f"{path.stem}.partial.wav")
        sf.write(partial, audio, TARGET_SAMPLE_RATE_HZ, subtype=TARGET_PCM_SUBTYPE)
        partial.replace(path)

    def build(
        self,
        output_root: Path,
        rebuild_labels: set[str] | None = None,
    ) -> DatasetManifest:
        output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        targets = {key: int(value) for key, value in dict(self.config["targets"]).items()}
        rebuild = frozenset(rebuild_labels or ())
        unknown_labels = rebuild - set(targets)
        if unknown_labels:
            raise ValueError(f"Unknown rebuild labels: {sorted(unknown_labels)}")
        state = {
            "config_hash": _json_hash(self.config),
            "source_manifest_hashes": self.source_manifest_hashes,
        }
        state_path = output_root / "build_state.json"
        previous_state = (
            json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        )
        if previous_state is not None and previous_state != state:
            if not rebuild:
                raise RuntimeError("Existing build_state.json does not match config/source manifests")
            manifest_path = output_root / "DatasetManifest.json"
            if not manifest_path.exists():
                raise RuntimeError("Targeted rebuild requires an existing complete manifest")
            previous_manifest = DatasetManifest.load(manifest_path)
            previous_config = dict((previous_manifest.generator or {}).get("config", {}))
            if _structural_config(previous_config) != _structural_config(self.config):
                raise RuntimeError("Targeted rebuild cannot change the structural dataset config")

        ratios = {key: float(value) for key, value in dict(self.config["split_ratios"]).items()}
        expected = sum(targets.values())
        records: list[DatasetRecord] = []
        completed = 0
        for label, total in targets.items():
            for split, count in split_counts(total, ratios).items():
                candidates = [
                    row
                    for row in self.sources
                    if row["label"] == label and row["split"] == split
                ]
                if label != "ambient" and not candidates:
                    raise ValueError(f"No {label} sources available for split {split}")
                for index in range(count):
                    record_id = f"{split}-{label}-{index:06d}"
                    path = output_root / split / label / f"{record_id}.wav"
                    if label == "ambient":
                        rng = np.random.default_rng(_stable_seed(f"{self.config['seed']}:{record_id}"))
                        categories = [item for item in self.config["noise_categories"] if item != "clean"]
                        category = str(categories[index % len(categories)])
                        duration = float(dict(self.config["mean_duration_seconds"])["ambient"])
                        audio = _procedural_noise(
                            category, int(duration * TARGET_SAMPLE_RATE_HZ), rng
                        )
                        peak = float(np.max(np.abs(audio)) + 1e-12)
                        audio = np.asarray(audio / peak * 0.35, dtype=np.float32)
                        acoustic = AcousticMetadata(noise_id=f"procedural_{category}")
                        source = {
                            "text": None,
                            "speaker_id": "none",
                            "source_family": "procedural_ambient_proxy",
                            "source_group_id": f"ambient:{split}",
                            "record_id": f"ambient:{split}:{index:06d}",
                            "gender": None,
                            "age_group": None,
                            "age_source": "unknown",
                            "hard_negative_tier": None,
                            "speed": None,
                        }
                    else:
                        source = candidates[index % len(candidates)]
                        audio = self._load_source(source)
                        audio, acoustic = self._augment(audio, split, label, index, record_id)
                        acoustic.speaking_rate = (
                            None if source.get("speed") is None else float(source["speed"])
                        )
                    self._write(path, audio, overwrite=label in rebuild)
                    info = sf.info(path)
                    records.append(
                        DatasetRecord(
                            record_id=record_id,
                            audio_path=path.relative_to(output_root).as_posix(),
                            label=label,
                            split=split,
                            text=source.get("text"),
                            speaker=SpeakerMetadata(
                                speaker_id=str(source["speaker_id"]),
                                source=str(source["source_family"]),
                                gender=source.get("gender"),
                                age_group=source.get("age_group"),
                                age_source=str(source.get("age_source", "unknown")),
                            ),
                            acoustic=acoustic,
                            sample_rate_hz=int(info.samplerate),
                            original_sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
                            duration_seconds=float(info.duration),
                            sha256=sha256_file(path),
                            source_utterance_id=str(source["record_id"]),
                            source_group_id=str(source["source_group_id"]),
                            augmentation_id=f"augmentation-{index:06d}",
                            hard_negative_tier=(
                                None
                                if source.get("hard_negative_tier") is None
                                else int(source["hard_negative_tier"])
                            ),
                        )
                    )
                    completed += 1
                    if completed % 250 == 0 or completed == expected:
                        partial_manifest = DatasetManifest(
                            wake_word=str(self.config["wake_word"]),
                            records=records,
                            source_kind="mixed",
                            root=str(output_root),
                            generator={"config": self.config, "state": state},
                        )
                        partial_manifest.save(output_root / "DatasetManifest.partial.json")
                        print(f"records={completed}/{expected} label={label} split={split}", flush=True)

        manifest = DatasetManifest(
            wake_word=str(self.config["wake_word"]),
            records=records,
            source_kind="mixed",
            root=str(output_root),
            generator={"config": self.config, "state": state},
        )
        manifest.save(output_root / "DatasetManifest.json")
        state_partial = output_root / "build_state.partial.json"
        state_partial.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        state_partial.replace(state_path)
        (output_root / "DatasetManifest.partial.json").unlink(missing_ok=True)
        return manifest
