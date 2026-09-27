"""HF source adapter. Import datasets only when an HF command actually runs."""

import json
from typing import Any, Iterator


FIELD_ALIASES = {"title": ("title", "name"), "text": ("text", "content", "article"),
                 "id": ("id", "page_id", "wikidata_id"), "url": ("url", "page_url", "link")}


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


def load_source(config: dict[str, Any]):
    from datasets import load_dataset

    kwargs = {"path": config["dataset_id"], "name": config.get("dataset_config"),
              "revision": config.get("dataset_revision"), "split": config["split"],
              "streaming": config["streaming"], "cache_dir": config.get("cache_dir")}
    return load_dataset(**kwargs)


def inspect(config: dict[str, Any], limit: int = 3) -> dict[str, Any]:
    dataset = load_source({**config, "streaming": True})
    names = list(dataset.features or {})
    iterator = iter(dataset)
    rows = []
    for _ in range(limit):
        try:
            rows.append(next(iterator))
        except StopIteration:
            break
    if not names and rows:
        names = list(rows[0])
    fields = detect_fields(names, config.get("fields", {}))
    return {"features": names, "field_mapping": fields, "examples": rows,
            "fingerprint": getattr(dataset, "_fingerprint", None)}


def iter_records(dataset) -> Iterator[tuple[int, dict[str, Any]]]:
    for index, row in enumerate(dataset):
        if isinstance(row, dict):
            yield index, row
