"""Extensible data-provider registry used by the product UI."""

from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import resolve_project_path


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    multi_speaker: bool = False
    gender_metadata: bool = False
    age_metadata: bool = False
    local_audio_import: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "multi_speaker": self.multi_speaker,
            "gender_metadata": self.gender_metadata,
            "age_metadata": self.age_metadata,
            "local_audio_import": self.local_audio_import,
        }


class DataProvider(ABC):
    id: str
    name: str

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_positive(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def generate_negative(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RegisteredProvider(DataProvider):
    id: str
    name: str
    kind: str
    description: str
    project_root: Path
    dependency: str | None = None
    configured_python: str | None = None
    worker: str | None = None
    capabilities: ProviderCapabilities = ProviderCapabilities()

    def python_executable(self) -> Path:
        if not self.dependency or importlib.util.find_spec(self.dependency) is not None:
            return Path(sys.executable).resolve()
        if self.configured_python:
            candidate = resolve_project_path(self.project_root, self.configured_python)
            if candidate.is_file():
                return candidate
        return Path(sys.executable).resolve()

    def available(self) -> bool:
        if not self.dependency:
            return True
        executable = self.python_executable()
        if executable == Path(sys.executable).resolve():
            return importlib.util.find_spec(self.dependency) is not None
        return executable.is_file()

    def generate_positive(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Provider generation is executed by the registered worker")

    def generate_negative(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Provider generation is executed by the registered worker")

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "available": self.available(),
            "python": str(self.python_executable()),
            "worker": self.worker,
            "capabilities": self.capabilities.as_dict(),
            "age_coverage_claimed": self.capabilities.age_metadata,
        }


class ProviderRegistry:
    def __init__(self, providers: list[RegisteredProvider]):
        self._by_name = {provider.name: provider for provider in providers}

    @classmethod
    def from_config(cls, project_root: Path, config: dict[str, Any]) -> "ProviderRegistry":
        providers = []
        for provider_id, raw in config.get("providers", {}).items():
            providers.append(
                RegisteredProvider(
                    id=str(provider_id),
                    name=str(raw["display_name"]),
                    kind=str(raw["kind"]),
                    description=str(raw.get("description", "")),
                    project_root=project_root.resolve(),
                    dependency=(str(raw["dependency"]) if raw.get("dependency") else None),
                    configured_python=(str(raw["python"]) if raw.get("python") else None),
                    worker=(str(raw["worker"]) if raw.get("worker") else None),
                    capabilities=ProviderCapabilities(**dict(raw.get("capabilities", {}))),
                )
            )
        return cls(providers)

    def by_name(self, name: str) -> RegisteredProvider:
        return self._by_name[name]

    def available(self, *, generation_only: bool = False) -> tuple[RegisteredProvider, ...]:
        values = tuple(provider for provider in self._by_name.values() if provider.available())
        if generation_only:
            values = tuple(
                provider for provider in values
                if provider.kind == "tts" or provider.capabilities.local_audio_import
            )
        return values
