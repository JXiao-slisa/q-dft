"""crewai 可选兼容层。

核心引擎/工具层（surface / mlip / dft / postprocess / report）在 mock 模式或
被 API 服务以子进程方式驱动时，并不需要完整的 crewai 运行时。这里在
crewai 缺失时提供一个保 ``.func`` 语义的 no-op ``@tool`` 装饰器，使工具
模块可以在仅有 ase/numpy 的精简环境中导入（CI-lite、边缘部署、单元测试）。

crewai 存在时行为与原生 ``@tool`` 完全一致。
"""

from __future__ import annotations

try:  # crewAI >= 1.x exposes the decorator at crewai.tools
    from crewai.tools import tool  # type: ignore
except Exception:  # pragma: no cover - exercised only without crewai
    def tool(func=None, *args, **kwargs):
        """Fallback decorator: identity function exposing ``.func``."""
        def wrap(f):
            f.func = f  # match crewai StructuredTool.func used by callers
            return f
        return wrap(func) if func is not None else wrap


__all__ = ["tool"]
