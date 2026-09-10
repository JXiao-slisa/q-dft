# 贡献指南（community edition）

感谢关注 slisaDFT！社区版遵循 Apache-2.0 许可，欢迎任何形式的贡献。

## 开发环境

```bash
git clone <repo> && cd q-dft
pip install -e ".[dev,mlip]"
cp .env.example .env    # 填入 LLM Key；SLISADFT_ENGINE_MODE=mock 可离线开发
pytest                  # 全部测试应在 mock 模式下通过
ruff check src/slisadft
```

## 提交规范

* 一个 PR 聚焦一件事；新功能必须带 pytest 测试（mock 引擎即可覆盖）。
* 遵守 `PRINCIPLES.md`：**Rule Zero** — 任何数据必须可溯源到真实计算文件；
  mock/合成数据必须带 SYNTHETIC 标记，绝不冒充真实结果。
* 提交信息使用英文或中文均可，格式 `类型: 摘要`（如 `fix: ...` / `feat: ...`）。

## 报告问题

请附上：`qdft check-env` 输出、`.env` 中引擎模式（不要贴 Key！）、
最小复现步骤与相关日志。
