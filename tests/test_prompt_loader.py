from pathlib import Path

import pytest

from app.prompts import loader


@pytest.fixture
def empty_cache():
    loader.load_prompt.cache_clear()
    yield
    loader.load_prompt.cache_clear()


def test_all_resources_are_cached_and_independent_of_working_directory(tmp_path, monkeypatch, empty_cache):
    monkeypatch.chdir(tmp_path)
    loader.validate_prompt_resources()
    expected = {name: loader.load_prompt(name) for name in loader.PROMPT_NAMES}
    assert len(expected) == len(set(loader.PROMPT_NAMES)) == 21
    assert {p.relative_to(loader.PROMPT_ROOT).with_suffix("").as_posix()
            for p in loader.PROMPT_ROOT.rglob("*.md")} == set(expected)
    def no_io(*args, **kwargs):
        raise AssertionError("Prompt loading reached the request hot path")
    monkeypatch.setattr(Path, "read_bytes", no_io)
    loader.validate_prompt_resources()
    assert {name: loader.load_prompt(name) for name in loader.PROMPT_NAMES} == expected


@pytest.mark.parametrize("name", ["../config", "/etc/passwd", "central/../planner", "unknown"])
def test_unknown_and_arbitrary_paths_rejected(name):
    with pytest.raises(loader.PromptResourceError, match="Unknown prompt resource"):
        loader.load_prompt(name)


def test_missing_resource_fails_clearly(tmp_path, monkeypatch, empty_cache):
    monkeypatch.setattr(loader, "PROMPT_ROOT", tmp_path)
    with pytest.raises(loader.PromptResourceError, match="Cannot load.*shared/grounded_system"):
        loader.validate_prompt_resources()


@pytest.mark.parametrize("content", [b"", b"  \n", b"\xff", b"hello\x00", b"\xef\xbb\xbfhello", b"hello\r\n"])
def test_malformed_prompt_rejected(tmp_path, monkeypatch, empty_cache, content):
    path = tmp_path / "central/planner.md"
    path.parent.mkdir()
    path.write_bytes(content)
    monkeypatch.setattr(loader, "PROMPT_ROOT", tmp_path)
    with pytest.raises(loader.PromptResourceError, match="(Cannot load|Malformed).*central/planner"):
        loader.load_prompt("central/planner")


def test_startup_validates_resources_before_loading_any_runtime(monkeypatch):
    import asyncio
    from app import main

    def invalid_resources():
        raise loader.PromptResourceError("Missing resource")
    def no_runtime():
        pytest.fail("Resources must be checked before constructing RAGService")
    monkeypatch.setattr(main, "validate_prompt_resources", invalid_resources)
    monkeypatch.setattr(main, "RAGService", no_runtime)
    async def start():
        async with main.lifespan(main.app):
            pytest.fail("Invalid resources must prevent startup")
    with pytest.raises(loader.PromptResourceError, match="Missing resource"):
        asyncio.run(start())
