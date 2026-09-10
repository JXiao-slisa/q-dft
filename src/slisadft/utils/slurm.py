"""SLURM job management utilities.

Handles job submission, monitoring, and result collection for DFT
calculations (VASP / CP2K / ABACUS) on HPC clusters with SLURM.
All environment-specific settings are read from environment variables
so the code is portable across machines.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SlurmConfig:
    """SLURM submission configuration.

    All values can be overridden by environment variables:
      SLURM_PARTITION, SLURM_NTASKS, SLURM_MEM, SLURM_TIME,
      MPI_RUN (mpirun/srun command or full path)
    """

    partition: str = os.environ.get("SLURM_PARTITION", "CPU")
    ntasks: int = int(os.environ.get("SLURM_NTASKS", "16"))
    mem_mb: int = int(os.environ.get("SLURM_MEM_MB", "6000"))
    time_limit: str = os.environ.get("SLURM_TIME", "24:00:00")
    mpi_cmd: str = os.environ.get("MPI_RUN", "mpirun")
    # If set, a `source <file>` line is prepended to the job script
    # (e.g. one-api environment setup before mpirun).
    env_script: Optional[str] = os.environ.get("MPI_ENV_SCRIPT") or None
    # Extra PATH entries (e.g. the dir containing vasp_std) as a ":"-joined string.
    extra_path: str = os.environ.get("MPI_EXTRA_PATH", "")

    def slurm_available(self) -> bool:
        """Return True if the `sbatch`/`squeue`/`sacct` commands exist."""
        return shutil.which("sbatch") is not None

    def _header(self, job_name: str, workdir: Path, out_prefix: str) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH -n {self.ntasks}",
            f"#SBATCH --output={out_prefix}.out",
            f"#SBATCH --error={out_prefix}.err",
            f"#SBATCH -t {self.time_limit}",
            f"#SBATCH --mem={self.mem_mb}",
            f"#SBATCH --chdir={workdir}",
            "",
        ]
        if self.env_script:
            lines.append(f"source {self.env_script}")
        if self.extra_path:
            lines.append(f"export PATH={self.extra_path}:${{PATH}}")
        lines.append("")
        return "\n".join(lines)

    def submit(self, job_name: str, workdir: Path, run_command: str,
               out_prefix: str = "slurm") -> Optional[int]:
        """Write a job script into ``workdir`` and submit it.

        Falls back to running the command locally (foreground) when SLURM
        is not available.  Returns the SLURM job id or None when the job
        was run locally.
        """
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        script = workdir / f"{out_prefix}.sh"
        body = self._header(job_name, workdir, out_prefix)
        body += run_command + "\n"
        script.write_text(body, encoding="utf-8")
        os.chmod(script, 0o755)

        if not self.slurm_available():
            # Local fallback: run the command detached in background.
            log = workdir / f"{out_prefix}.out"
            err = workdir / f"{out_prefix}.err"
            with open(log, "a") as out_f, open(err, "a") as err_f:
                proc = subprocess.Popen(
                    ["bash", str(script)],
                    stdout=out_f,
                    stderr=err_f,
                    cwd=str(workdir),
                )
                return proc.pid

        proc = subprocess.run(
            ["sbatch", str(script)],
            capture_output=True, text=True, cwd=str(workdir), timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {proc.stderr}")
        m = re.search(r"Submitted batch job (\d+)", proc.stdout)
        if not m:
            raise RuntimeError(f"Unparsable sbatch output: {proc.stdout}")
        return int(m.group(1))


def job_status(job_id: int) -> str:
    """Return SLURM job state (PENDING/RUNNING/COMPLETED/FAILED/...) or
    'UNKNOWN' when squeue/sacct cannot be queried."""
    try:
        proc = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True, text=True, timeout=30,
        )
        state = proc.stdout.strip()
        if state:
            return state
    except Exception:
        pass
    # Fall back to sacct for finished jobs.
    try:
        proc = subprocess.run(
            ["sacct", "-j", str(job_id), "-n", "-X", "-o", "State"],
            capture_output=True, text=True, timeout=30,
        )
        state = proc.stdout.strip().splitlines()
        if state:
            return state[0].split()[0]
    except Exception:
        pass
    return "UNKNOWN"


def wait_job(job_id: int, poll_interval: float = 30.0,
             timeout: Optional[float] = None) -> str:
    """Block until ``job_id`` reaches a terminal state.

    Returns the final state string.  Raises TimeoutError when ``timeout``
    seconds elapse first.
    """
    start = time.time()
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
    while True:
        state = job_status(job_id)
        if state in terminal:
            return state
        if state == "UNKNOWN":
            # Job may have finished and left the queue; check completion files.
            return state
        if timeout is not None and time.time() - start > timeout:
            raise TimeoutError(f"job {job_id} still {state} after {timeout}s")
        time.sleep(poll_interval)


def detect_mpi_env() -> dict:
    """Return a description of the compute environment (for logging/reports)."""
    info = {
        "nproc": os.cpu_count(),
        "sbatch": shutil.which("sbatch"),
        "mpirun": shutil.which("mpirun"),
    }
    for key in ("SLURM_PARTITION", "SLURM_NTASKS", "VASP_PATH",
                "VASPKIT_PATH", "CP2K_BIN", "ABACUS_BIN"):
        info[key] = os.environ.get(key, "")
    return info