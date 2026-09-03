"""Import and dependency gates; no model construction."""
import ast
import importlib
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def imports(path):
    module = ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = ".".join(package.split(".")[:len(package.split(".")) - node.level + 1] + ([name] if name else []))
            yield name
            yield from (name + "." + alias.name for alias in node.names)


def test_public_roles_resolve_to_single_canonical_implementations():
    for package, symbol in (("research", "ResearchAgent"), ("evidence", "EvidenceCriticAgent"),
                            ("history_answerer", "HistoryAnswererAgent"), ("central", "CentralAgent")):
        public = importlib.import_module("app.agents." + package)
        canonical = importlib.import_module("app.agents." + package + ".agent")
        assert getattr(public, symbol) is getattr(canonical, symbol)
        typing.get_type_hints(getattr(public, symbol).__init__)
    from app.agents.three_llm import AgentOrchestrator
    from app.agents.orchestrator import AgentOrchestrator as compatibility
    assert AgentOrchestrator is compatibility
    from app.agents.hybrid import HybridRAGOrchestrator
    typing.get_type_hints(HybridRAGOrchestrator.__init__)
    names = {"ResearchAgent", "EvidenceCriticAgent", "HistoryAnswererAgent", "CentralAgent", "AgentOrchestrator",
             "HybridRAGOrchestrator", "CentralAgentConfig", "AgentConfig", "EvidenceChunk"}
    counts = dict.fromkeys(names, 0)
    for path in (ROOT / "app/agents").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef) and node.name in names:
                counts[node.name] += 1
    assert set(counts.values()) == {1}, counts


def test_dependency_direction_and_no_cycles_even_for_deferred_imports():
    graph = {}
    for folder in ("app", "training", "evaluation"):
        for path in (ROOT / folder).rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            dependencies = set(imports(path))
            if folder == "app":
                assert not any(dep.startswith(("training", "evaluation")) for dep in dependencies), relative
            if folder == "training":
                assert not any(dep.startswith("evaluation") for dep in dependencies), relative
            if folder == "evaluation":
                assert not any(dep.startswith("training") for dep in dependencies), relative
            if relative.startswith("app/agents/central/"):
                assert not any(dep.startswith(tuple("app.agents." + name for name in ("research", "evidence", "history_answerer", "three_llm", "orchestrator", "hybrid"))) for dep in dependencies), relative
            if relative.startswith("app/agents/common/"):
                assert not any(dep.startswith("app.agents.") and not dep.startswith("app.agents.common.") for dep in dependencies), relative
            if relative.startswith("app/agents/"):
                module = relative.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
                graph[module] = dependencies
    done, active = set(), []
    def visit(name):
        assert name not in active, " -> ".join([*active, name])
        if name in done:
            return
        active.append(name)
        for dependency in graph[name] & graph.keys():
            visit(dependency)
        active.pop()
        done.add(name)
    for name in graph:
        visit(name)


def test_flat_implementations_removed_and_shims_are_reexports_only():
    agents = ROOT / "app/agents"
    assert not list(agents.glob("central_*.py"))
    for name in ("research_agent.py", "evidence_agent.py", "history_answerer.py", "model_runtime.py"):
        assert not (agents / name).exists()
    for name in ("config.py", "schemas.py", "prompts.py", "orchestrator.py"):
        tree = ast.parse((agents / name).read_text(encoding="utf-8"))
        assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
