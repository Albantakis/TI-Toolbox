"""TI Exhaustive Search Module."""

from tit.opt.config import ExConfig, ExResult

__all__ = [
    "ExConfig",
    "ExResult",
    "ExSearchEngine",
    "run_ex_search",
]


def __getattr__(name):
    if name == "ExSearchEngine":
        from tit.opt.ex.engine import ExSearchEngine

        return ExSearchEngine
    if name == "run_ex_search":
        from tit.opt.ex.ex import run_ex_search

        return run_ex_search
    raise AttributeError(name)
