from __future__ import annotations

from app.agents.lazy_runtime import LazyRuntime


def test_mode_runtimes_initialize_only_when_the_selected_proxy_is_used():
    loads: list[str] = []

    def factory(name):
        def build():
            loads.append(name)
            return type("Runtime", (), {"name": name})()
        return build

    role = LazyRuntime(factory("qwen3-4b"), name="roles")
    central = LazyRuntime(factory("qwen3-8b"), name="central")

    assert not role.loaded and not central.loaded
    assert central.name == "qwen3-8b"
    assert loads == ["qwen3-8b"]
    assert not role.loaded and central.loaded
    assert role.name == "qwen3-4b"
    assert loads == ["qwen3-8b", "qwen3-4b"]

