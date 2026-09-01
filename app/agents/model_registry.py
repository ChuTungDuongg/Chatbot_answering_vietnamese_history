from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


SHARED_BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
CENTRAL_BASE_MODEL_ID = "Qwen/Qwen3-8B"
CENTRAL_ADAPTER_PATH = "adapters/central"
RoleName = Literal["research", "evidence", "history"]


@dataclass(frozen=True)
class RoleModelSpec:
    role: RoleName
    model_name: str
    adapter_path: str
    expected_base_model_id: str
    generation: dict[str, Any]


@dataclass(frozen=True)
class CentralModelSpec:
    model_name: str
    adapter_path: str
    expected_base_model_id: str
    generation: dict[str, Any]


CENTRAL_MODEL = CentralModelSpec(
    model_name="central",
    adapter_path=CENTRAL_ADAPTER_PATH,
    expected_base_model_id=CENTRAL_BASE_MODEL_ID,
    generation={"max_new_tokens": 1536, "temperature": 0.0, "top_p": 1.0},
)


ROLE_MODELS: dict[RoleName, RoleModelSpec] = {
    "research": RoleModelSpec(
        role="research",
        model_name="research",
        adapter_path="adapters/research",
        expected_base_model_id=SHARED_BASE_MODEL_ID,
        generation={"max_new_tokens": 256, "temperature": 0.0, "top_p": 1.0},
    ),
    "evidence": RoleModelSpec(
        role="evidence",
        model_name="evidence",
        adapter_path="adapters/evidence",
        expected_base_model_id=SHARED_BASE_MODEL_ID,
        generation={"max_new_tokens": 640, "temperature": 0.0, "top_p": 1.0},
    ),
    "history": RoleModelSpec(
        role="history",
        model_name="history",
        adapter_path="adapters/history",
        expected_base_model_id=SHARED_BASE_MODEL_ID,
        generation={"max_new_tokens": 1536, "temperature": 0.0, "top_p": 1.0},
    ),
}

def registry_manifest() -> dict[str, Any]:
    return {
        "shared_base_model_id": SHARED_BASE_MODEL_ID,
        "tokenizer_model_id": SHARED_BASE_MODEL_ID,
        "roles": {name: asdict(spec) for name, spec in ROLE_MODELS.items()},
        "legacy_models": {},
        "central": asdict(CENTRAL_MODEL),
    }


def adapter_declared_base(adapter_path: str | Path) -> str:
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"adapter config not found: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid adapter config {config_path}: {exc}") from exc
    base = str(payload.get("base_model_name_or_path") or "").strip()
    if not base:
        raise ValueError(f"adapter config does not declare base_model_name_or_path: {config_path}")
    return base


def validate_role_adapter(role: RoleName, adapter_path: str | Path) -> str:
    if role not in ROLE_MODELS:
        raise ValueError(f"unknown active model role: {role!r}")
    actual = adapter_declared_base(adapter_path)
    expected = ROLE_MODELS[role].expected_base_model_id
    if actual != expected:
        raise ValueError(
            f"{role} adapter/base mismatch: expected {expected!r}, found {actual!r}"
        )
    return actual


def validate_central_adapter(adapter_path: str | Path) -> str:
    actual = adapter_declared_base(adapter_path)
    expected = CENTRAL_MODEL.expected_base_model_id
    if actual != expected:
        raise ValueError(
            f"central adapter/base mismatch: expected {expected!r}, found {actual!r}"
        )
    return actual


def validate_active_role_names() -> None:
    names = [spec.model_name for spec in ROLE_MODELS.values()]
    if len(names) != len(set(names)):
        raise ValueError("active role model names must be unique")
    if any(spec.expected_base_model_id != SHARED_BASE_MODEL_ID for spec in ROLE_MODELS.values()):
        raise ValueError("all active role adapters must target the canonical shared base")


validate_active_role_names()
