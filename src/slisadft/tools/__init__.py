"""slisaDFT tools package.

All CrewAI tools are organized by domain:
  - surface_tools:     model building (slab, adsorption, molecule)
  - mlip_tools:        MACE / DPA-4 optimization
  - dft_tools:         VASP / CP2K / ABACUS execution
  - postprocess_tools: adsorption energy, free energy, DOS, d-band
  - report_tools:      report generation

Import tools directly from the submodules:
    from .tools.surface_tools import build_surface_tool
"""

from .surface_tools import (
    build_surface_tool,
    build_adsorption_tool,
    build_molecule_tool,
    build_slab_with_adsorbate_tool,
)

from .mlip_tools import (
    mace_optimize_tool,
    dpa_optimize_tool,
    mlip_single_point_tool,
)

from .dft_tools import (
    dft_optimize_tool,
    dft_single_point_tool,
    check_dft_environment_tool,
)

from .postprocess_tools import (
    adsorption_energy_tool,
    gibbs_free_energy_tool,
    reaction_step_diagram_tool,
    analyze_dos_tool,
    generate_volcano_plot_tool,
)

from .report_tools import (
    generate_report_tool,
    collect_results_tool,
    finalize_report_tool,
)

__all__ = [
    "build_surface_tool",
    "build_adsorption_tool",
    "build_molecule_tool",
    "build_slab_with_adsorbate_tool",
    "mace_optimize_tool",
    "dpa_optimize_tool",
    "mlip_single_point_tool",
    "dft_optimize_tool",
    "dft_single_point_tool",
    "check_dft_environment_tool",
    "adsorption_energy_tool",
    "gibbs_free_energy_tool",
    "reaction_step_diagram_tool",
    "analyze_dos_tool",
    "generate_volcano_plot_tool",
    "generate_report_tool",
    "collect_results_tool",
    "finalize_report_tool",
]