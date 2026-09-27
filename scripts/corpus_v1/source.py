"""Resolve immutable Hugging Face source metadata and load selected splits."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote


FIELD_ALIASES = {"title": ("title", "name"), "text": ("text", "content", "article"),
                 "id": ("id", "page_id", "wikidata_id"), "url": ("url", "page_url", "link")}
PRESETS = {"uvw-2026": Path(__file__).resolve().parents[2] / "configs/corpus_v1/uvw_2026.json"}
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def preset(name: str) -> dict[str, Any]:
    try:
        data = json.loads(PRESETS[name].read_text(encoding="utf-8"))
    except KeyError as exc:
        raise ValueError(f"Unknown Corpus V1 preset: {name}") from exc
    if data.get("preset_version") != 1:
        raise ValueError(f"Unsupported preset version for {name}")
    return data


def detect_fields(names: list[str], overrides: dict[str, str | None]) -> dict[str, str | None]:
    fields = {}
    for key, aliases in FIELD_ALIASES.items():
        chosen = overrides.get(key) or next((name for name in aliases if name in names), None)
        if chosen and chosen not in names:
            raise ValueError(f"Field {chosen!r} for {key} is absent; available: {names}")
        fields[key] = chosen
    if not fields["title"] or not fields["text"]:
        raise ValueError(f"Cannot detect title/text fields: {names}; pass explicit field flags")
    return fields


def resolve_source(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve requested ref to a Hub commit; failure never falls back to a mutable ref."""
    from huggingface_hub import HfApi

    requested = config.get("requested_revision") or "main"
    dataset_id = config["dataset_id"]
    try:
        info = HfApi().dataset_info(dataset_id, revision=requested)
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve Hugging Face dataset {dataset_id}@{requested}: {exc}") from exc
    sha = str(info.sha or "").lower()
    if not SHA_PATTERN.fullmatch(sha):
        raise RuntimeError(f"Hugging Face did not return an immutable commit SHA for {dataset_id}@{requested}")
    card = info.card_data.to_dict() if info.card_data else {}
    dataset_info = card.get("dataset_info") or {}
    if isinstance(dataset_info, list):
        dataset_info = next((x for x in dataset_info if x.get("config_name") == config.get("dataset_config")),
                            dataset_info[0] if dataset_info else {})
    features = [item["name"] for item in dataset_info.get("features", []) if "name" in item]
    sizes = {item["name"]: int(item["num_examples"]) for item in dataset_info.get("splits", [])
             if item.get("name") and item.get("num_examples") is not None}
    configs = [item["config_name"] for item in card.get("configs", []) if item.get("config_name")]
    source_url = f"https://huggingface.co/datasets/{dataset_id}"
    return {"dataset_id": dataset_id, "dataset_config": config.get("dataset_config"),
            "requested_revision": requested, "resolved_revision_sha": sha,
            "available_splits": list(sizes), "expected_split_sizes": sizes,
            "features": features, "configs": configs,
            "license": card.get("license"), "language": card.get("language"),
            "source_url": source_url, "dataset_card_url": source_url + "/blob/" + sha + "/README.md",
            "card_metadata": {key: card.get(key) for key in
                              ("pretty_name", "source_datasets", "tags", "size_categories")},
            "hf_fingerprint": None}


def selected_splits(config: dict[str, Any], source_info: dict[str, Any]) -> list[str]:
    requested = config.get("split", "train")
    available = source_info.get("available_splits") or []
    if requested == "all":
        if not available:
            raise ValueError("--split all requires source split metadata; specify --split names explicitly")
        return list(available)
    splits = [item.strip() for item in requested.split(",") if item.strip()]
    if not splits or len(splits) != len(set(splits)):
        raise ValueError("Provide one or more unique source splits")
    if available and any(item not in available for item in splits):
        raise ValueError(f"Requested split absent from source: {splits}; available: {available}")
    return splits


def load_source(config: dict[str, Any], split: str):
    from datasets import load_dataset

    sha = config.get("resolved_revision_sha")
    if not SHA_PATTERN.fullmatch(str(sha or "")):
        raise ValueError("Dataset loading requires resolved_revision_sha")
    return load_dataset(path=config["dataset_id"], name=config.get("dataset_config"),
                        revision=sha, split=split, streaming=config.get("streaming", True),
                        cache_dir=config.get("cache_dir"))


def inspect(config: dict[str, Any], limit: int = 3,
            source_info: dict[str, Any] | None = None, loader=None) -> dict[str, Any]:
    info = source_info or resolve_source(config)
    splits = selected_splits(config, info)
    pinned = {**config, "resolved_revision_sha": info["resolved_revision_sha"]}
    examples = {}
    observed_names = list(info.get("features") or [])
    for split in splits:
        dataset = loader(split) if loader else load_source({**pinned, "streaming": True}, split)
        rows = []
        for row in dataset:
            preview = {}
            for key, value in row.items():
                preview[key] = value[:240] + ("…" if len(value) > 240 else "") if isinstance(value, str) else value
            rows.append(preview)
            if len(rows) >= limit:
                break
        examples[split] = rows
        if not observed_names and rows:
            observed_names = list(rows[0])
    fields = detect_fields(observed_names, config.get("fields", {}))
    return {**info, "source_splits": splits, "field_mapping": fields, "examples": examples}


def article_url(raw: dict[str, Any], fields: dict[str, str | None], source_kind: str) -> str:
    explicit = str(raw.get(fields["url"], "") or "") if fields["url"] else ""
    if explicit:
        return explicit
    article_id = str(raw.get(fields["id"], "") or "") if fields["id"] else ""
    if source_kind == "uvw-2026" and article_id:
        # The UVW card defines `id` as the URL-safe Wikipedia article title.
        return "https://vi.wikipedia.org/wiki/" + quote(article_id.replace(" ", "_"), safe="()_")
    title = str(raw.get(fields["title"], "") or "")
    return "https://vi.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_") if title else ""
