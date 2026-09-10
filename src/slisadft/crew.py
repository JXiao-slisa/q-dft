import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from dotenv import load_dotenv

from .tools.surface_tools import (
    build_surface_tool,
    build_adsorption_tool,
    build_molecule_tool,
    build_slab_with_adsorbate_tool,
)
from .tools.mlip_tools import (
    mace_optimize_tool,
    dpa_optimize_tool,
    mlip_single_point_tool,
)
from .tools.dft_tools import (
    dft_optimize_tool,
    dft_single_point_tool,
    check_dft_environment_tool,
)
from .tools.postprocess_tools import (
    adsorption_energy_tool,
    gibbs_free_energy_tool,
    reaction_step_diagram_tool,
    vibrational_thermochemistry_tool,
    analyze_dos_tool,
    generate_volcano_plot_tool,
)
from .tools.report_tools import (
    generate_report_tool,
    collect_results_tool,
    finalize_report_tool,
)

load_dotenv()

# Also load the project-root .env (repo checkout layout) so that processes
# started with a different cwd (e.g. API job runners) still pick up config.
from pathlib import Path as _Path

_PROJECT_ENV = _Path(__file__).resolve().parents[2] / ".env"
if _PROJECT_ENV.is_file():
    load_dotenv(_PROJECT_ENV, override=False)


def _resolve_llm_kwargs() -> dict:
    """Build LLM constructor kwargs from environment configuration.

    Supported env vars (see .env.example):
      LLM_BASE_URL / LLM_MODEL / LLM_API_KEY — any OpenAI-compatible endpoint
      (DeepSeek, SenseNova, Qwen, OpenAI, vLLM ...).  Falls back to the
      legacy DEEPSEEK_API_KEY / SENSENOVA_API_KEY variables when present.
    """
    model = os.getenv("LLM_MODEL", "deepseek-chat").strip()
    # litellm needs a provider prefix; "openai/..." + base_url selects any
    # OpenAI-compatible endpoint.
    if "/" not in model:
        model = f"openai/{model}"
    api_key = (os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
               or os.getenv("SENSENOVA_API_KEY") or "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").strip()
    return {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
    }


def prepare_inputs(inputs: dict) -> dict:
    """Normalize kickoff inputs: fill defaults and inject the knowledge digest.

    RSI P4: the reviewed-rules digest from ``knowledge/computational_rules.md``
    is injected as ``knowledge_context`` (referenced by tasks.yaml).  Missing
    keys would break crewAI placeholder interpolation, so every workflow
    input dict must pass through this function before ``kickoff``.
    """
    inputs = dict(inputs or {})
    defaults = {
        "element": "Pt",
        "miller": "(111)",
        "layers": 3,
        "adsorbate": "CO",
        "site": "top",
        "dft_calculator": "vasp",
    }
    for key, value in defaults.items():
        inputs.setdefault(key, value)
    if "knowledge_context" not in inputs:
        from .utils.knowledge_context import build_knowledge_context
        inputs["knowledge_context"] = build_knowledge_context()
    return inputs


@CrewBase
class Slisadft():
    agents: list[BaseAgent]
    tasks: list[Task]

    def __init__(self):
        # LLM endpoint is fully environment-configurable (.env), defaulting
        # to DeepSeek Chat via the OpenAI-compatible API.
        self.llm = LLM(**_resolve_llm_kwargs())

    # ── Agents ──────────────────────────────────────────────────────────

    @agent
    def surface_scientist(self) -> Agent:
        return Agent(
            config=self.agents_config['surface_scientist'],
            tools=[
                build_surface_tool,
                build_adsorption_tool,
                build_molecule_tool,
                build_slab_with_adsorbate_tool,
            ],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def mlip_optimizer(self) -> Agent:
        return Agent(
            config=self.agents_config['mlip_optimizer'],
            tools=[
                mace_optimize_tool,
                dpa_optimize_tool,
                mlip_single_point_tool,
            ],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def dft_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['dft_engineer'],
            tools=[
                dft_optimize_tool,
                dft_single_point_tool,
                check_dft_environment_tool,
            ],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def postprocess_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['postprocess_analyst'],
            tools=[
                adsorption_energy_tool,
                gibbs_free_energy_tool,
                reaction_step_diagram_tool,
                vibrational_thermochemistry_tool,
                analyze_dos_tool,
                generate_volcano_plot_tool,
            ],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['report_writer'],
            tools=[
                generate_report_tool,
                collect_results_tool,
                finalize_report_tool,
            ],
            llm=self.llm,
            verbose=True,
        )

    # ── Tasks ───────────────────────────────────────────────────────────

    @task
    def build_surface_task(self) -> Task:
        return Task(config=self.tasks_config['build_surface_task'])

    @task
    def mlip_optimize_task(self) -> Task:
        return Task(config=self.tasks_config['mlip_optimize_task'])

    @task
    def dft_optimize_task(self) -> Task:
        return Task(config=self.tasks_config['dft_optimize_task'])

    @task
    def calculate_adsorption_energy_task(self) -> Task:
        return Task(config=self.tasks_config['calculate_adsorption_energy_task'])

    @task
    def calculate_free_energy_task(self) -> Task:
        return Task(config=self.tasks_config['calculate_free_energy_task'])

    @task
    def analyze_electronic_structure_task(self) -> Task:
        return Task(config=self.tasks_config['analyze_electronic_structure_task'])

    @task
    def generate_report_task(self) -> Task:
        return Task(config=self.tasks_config['generate_report_task'])

    # ── Crews ───────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Default sequential crew: full DFT pipeline."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    # convenience: expose alternative crews as regular methods
    def adsorption_crew(self) -> Crew:
        """Adsorption energy workflow: build → MLIP optimize → DFT → adsorption energy."""
        return Crew(
            agents=[self.surface_scientist(), self.mlip_optimizer(),
                    self.dft_engineer(), self.postprocess_analyst()],
            tasks=[self.build_surface_task(), self.mlip_optimize_task(),
                   self.dft_optimize_task(), self.calculate_adsorption_energy_task()],
            process=Process.sequential,
            verbose=True,
        )

    def volcano_crew(self) -> Crew:
        """Free-energy diagram workflow: build → MLIP → DFT → free energy → plot."""
        return Crew(
            agents=[self.surface_scientist(), self.mlip_optimizer(),
                    self.dft_engineer(), self.postprocess_analyst()],
            tasks=[self.build_surface_task(), self.mlip_optimize_task(),
                   self.dft_optimize_task(), self.calculate_free_energy_task()],
            process=Process.sequential,
            verbose=True,
        )