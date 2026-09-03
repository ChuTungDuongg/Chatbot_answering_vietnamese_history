"""Public API; importing helpers does not initialize agent/model dependencies."""
from importlib import import_module

_EXPORTS = {'AgentOrchestrator': 'orchestrator'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
