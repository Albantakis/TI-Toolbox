"""Multipolar exhaustive-search public API."""

from tit.opt.config import MExConfig, MExResult

__all__ = ["MExConfig", "MExResult", "run_m_ex_search"]


def __getattr__(name):
    if name == "run_m_ex_search":
        from tit.opt.mex.mex import run_m_ex_search

        return run_m_ex_search
    raise AttributeError(name)
