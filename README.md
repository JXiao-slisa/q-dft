<div align="center">

# ⚛️ q-dft agent

### 🚀 Self-Evolving AI Multi-Agent System for Autonomous DFT Catalysis Research

**Quantum · Autonomous · Trustworthy**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![crewAI](https://img.shields.io/badge/multi--agent-crewAI-FF4F5B)](https://github.com/crewAIInc/crewAI)
[![Version](https://img.shields.io/badge/version-0.4.0-6E56CF)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-88%20passing-3FB950)](tests)
[![Mock Mode](https://img.shields.io/badge/offline-mock%20engine-8B5CF6)](#-30-second-quickstart)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-ff69b4)](CONTRIBUTING.md)

*Let LLM agents run your DFT workflows — with every number traceable to a real calculation.*

`Model` → `MLIP pre-optimize` → `DFT` → `Post-process` → `Report` — fully automated.

</div>

---

## ⚡ What is it

**q-dft agent** is an open-source (Apache-2.0) **AI multi-agent platform for
first-principles (DFT) computational catalysis research**, delivered as a
CLI / Python agent. A crew of five domain-expert agents autonomously runs the
complete pipeline — from surface model construction to publication-grade
research reports — while the **Rule Zero mechanism** guarantees it will never
fabricate a single number.

```text
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
  │ 🔬 Surface  │──▶│ ⚡ MLIP     │──▶│ 🧮 DFT      │──▶│ 📊 Postproc  │──▶│ 📝 Report   │
  │ Scientist   │   │ Optimizer   │   │ Engineer    │   │ Analyst      │   │ Writer      │
  │ slabs/sites │   │ MACE / DPA  │   │VASP/CP2K/   │   │E_ad/Gibbs/   │   │ full report │
  │             │   │             │   │ABACUS+SLURM │   │DOS/d-band    │   │             │
  └─────────────┘   └─────────────┘   └─────────────┘   └──────────────┘   └─────────────┘
        ▲                                                                        │
        └──────────── 🧠 RSI loop: experience → rules → context injection ◀──────┘
```

## 🔥 Highlights

| | Capability | Why it matters |
|---|---|---|
| 🛡️ | **Rule Zero — zero fabricated data** | Every energy is traceable to a real calculation file; mock data is force-stamped `SYNTHETIC`; failures are reported honestly, never dressed up |
| 🧠 | **RSI — recursive self-improvement** | Every run extracts structured experience → human review promotes it to a rule → rules are auto-injected into agent context. **The platform gets smarter as you use it** |
| 🎯 | **Deterministic protocol layer** | The three-point adsorption-energy protocol is hardened Python code, not LLM improvisation — numerical completeness never depends on prompt luck |
| 🔌 | **Real / Mock dual-mode engines** | VASP · CP2K · ABACUS + MACE · DPA-4; no engine installed? Mock mode runs the entire pipeline offline |
| 🧬 | **Failure intelligence** | 10 known DFT failure patterns auto-diagnosed with actionable fixes (SCF non-convergence? missing POTCAR? ZBRENT crash?) |
| 🔭 | **Hypothesis engine** | Suggests the next most informative calculation from results + knowledge base — the first piece of a closed research loop |
| 🚀 | **One-command workflow** | `qdft init` scaffolds a project, `qdft info` inspects structures, `qdft runs` browses your research history, `qdft cite` exports BibTeX |
| 🌐 | **Any LLM backend** | DeepSeek / Qwen / OpenAI / SenseNova / vLLM — switch with one environment variable |

## ⚙️ Installation

```bash
git clone https://github.com/JXiao-slisa/q-dft.git
cd q-dft
pip install -e .            # core
pip install -e ".[mlip]"    # real-computation mode (MACE / DPA-4; install PyTorch first)
pip install -e ".[dev]"     # development & tests
```

Requires Python ≥ 3.10, < 3.13 (3.11 recommended).

```bash
cp .env.example .env        # fill in LLM_API_KEY (any OpenAI-compatible endpoint)
```

## ⚡ 30-Second Quickstart

No VASP, no GPU, no cluster needed — run the full pipeline with the mock engine:

```bash
qdft check-env                                       # engine self-check
qdft adsorption-full --inputs '{"element": "Pt", "adsorbate": "CO"}'
# ✅ three-point protocol: build → MLIP → DFT ×3 → E_ad (12 s, SYNTHETIC-stamped)

qdft batch --inputs-file combos.json --out results.csv   # high-throughput screening
qdft suggest                                         # AI proposes the next hypothesis
```

> ⚠️ **Rule Zero**: mock-mode output is synthetic data (force-stamped) for demo
> and testing only — never use it as research data. For production, configure
> real engines ([deployment guide](docs/DEPLOYMENT.md)).

## 🖥️ Real Calculations (Cluster Mode)

```ini
# .env
SLISADFT_ENGINE_MODE=real
VASP_EXE=/opt/vasp/vasp.6.5.1/bin/vasp_std
VASP_POTCAR_DIR=/opt/vasp/potpaw_PBE
SLURM_PARTITION=CPU
MPI_RUN=srun
```

```bash
qdft run --inputs '{"element":"Pt","miller":"(111)","adsorbate":"CO","site":"top","dft_calculator":"vasp"}'
```

Full SLURM / open-source engine setup: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

## 🧠 RSI: An Agent That Evolves With Your Research

```text
calculation done ──▶ auto-extract experience (YAML) ──▶ pending_rules/ ──▶ human review
                              ▲                                            │
                              └── inject into agent context ◀── knowledge/ ┘
```

- **Auditable**: every rule keeps its source calculation and review record
- **Pluggable**: `{knowledge_context}` is auto-injected into the modeling/DFT agents
- **Programmable**: script reviews via `slisadft.utils.rsi_review.approve_rule(...)`

## 🧪 Quality Assurance

```bash
SLISADFT_ENGINE_MODE=mock pytest    # 60 tests, all green, fully offline
ruff check src/slisadft             # lint
```

Test coverage: VASP input/output parsing (incl. frequency INCAR + real force
tables), thermochemistry (ZPE / CHE), mock-engine determinism, three-point
protocol consistency, batch CSV, failure diagnosis, parity QA, knowledge
injection, rule review (incl. path-traversal guards), end-to-end workflows.

## 🗺️ Roadmap

- [x] Multi-agent pipeline + three workflows
- [x] Real/Mock dual-mode engines + SYNTHETIC stamping
- [x] RSI experience extraction / review flow / context injection / hypothesis engine
- [x] Deterministic three-point adsorption protocol · batch screening · failure intelligence · parity QA
- [ ] Full P6 hypothesis loop (suggest → execute → verify, autonomously)
- [ ] Active-learning-driven high-throughput electrocatalysis screening
- [ ] Pro edition: Web UI + REST API + job runner (commercial)

## 🤝 Contributing

PRs welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md). New features
should ship with tests that run in mock mode — protecting Rule Zero is
everyone's job.

## 📄 License

[Apache-2.0](LICENSE). VASP is third-party commercial software — users must
hold their own license; this project neither bundles nor distributes it
(see [LICENSES/THIRD_PARTY.md](LICENSES/THIRD_PARTY.md)).

---

<div align="center">

**q-dft agent** · *Trustworthy autonomous DFT research, one commit at a time.*

⚡ If this project helps you, please give it a star ⭐

</div>
