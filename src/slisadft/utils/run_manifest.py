"""运行溯源清单（Rule Zero 的工程支撑）。

每次工作流运行生成 ``run_manifest.json``，记录：工作流类型、输入、
引擎模式、LLM 模型、软件版本与关键输入文件哈希——让每个结果都能
回答"它是由什么输入、什么配置、哪个版本的代码产生的"。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def _sha256(path: Path, limit_bytes: int = 64 * 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def create_run_manifest(workflow: str, inputs: dict,
                        workdir: Optional[Path] = None) -> Dict[str, object]:
    """Build (and optionally write) a manifest for a workflow run."""
    from ..engines import engine_mode

    manifest: Dict[str, object] = {
        "schema": "slisadft.run_manifest/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow,
        "inputs": inputs,
        "engine_mode": engine_mode(),
        "llm_model": os.getenv("LLM_MODEL", "deepseek-chat"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    # Hash any input files referenced by path so inputs are tamper-evident.
    input_files: Dict[str, str] = {}
    for key, value in (inputs or {}).items():
        if isinstance(value, str) and value and Path(value).is_file():
            digest = _sha256(Path(value))
            if digest:
                input_files[value] = digest
    if input_files:
        manifest["input_file_hashes"] = input_files

    if workdir is not None:
        manifest_path = Path(workdir) / "run_manifest.json"
        try:
            Path(workdir).mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8")
            manifest["manifest_path"] = str(manifest_path)
        except OSError:
            pass
    return manifest


def finalize_run_manifest(manifest_path: Path, status: str,
                          result: Optional[dict] = None) -> Optional[Path]:
    """Record the final status/result of a run in its manifest."""
    path = Path(manifest_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data["finished_at"] = datetime.now(timezone.utc).isoformat()
    data["status"] = status
    if result is not None:
        data["result_summary"] = result
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except OSError:
        return None
    return path
