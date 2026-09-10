"""VASP input generation and output parsing (incl. frequency fix)."""

from slisadft.utils.vasp_input import (
    DEFAULT_INCAR,
    FREQ_INCAR_DEFAULTS,
    parse_vasp_converged,
    parse_vasp_energy,
    parse_vasp_frequencies,
    parse_vasp_forces,
    write_incar,
    write_kpoints,
)


def test_default_incar_relax(tmp_path):
    path = write_incar(tmp_path, relax=True)
    text = path.read_text()
    assert "IBRION = 2" in text
    assert "NSW = 100" in text


def test_default_incar_single_point_disables_ionic(tmp_path):
    path = write_incar(tmp_path, relax=False)
    text = path.read_text()
    assert "IBRION = -1" in text
    assert "NSW = 0" in text


def test_freq_incar_fixes_displacement_bug(tmp_path):
    """IBRION=5 with NSW=0 produces no displacement steps — the fix forces NSW>=1."""
    path = write_incar(tmp_path, relax=False, freq=True)
    text = path.read_text()
    assert "IBRION = 5" in text
    assert "NSW = 1" in text
    assert "NFREE = 2" in text
    assert "ISYM = 0" in text


def test_freq_incar_user_overrides_win(tmp_path):
    path = write_incar(tmp_path, settings={"POTIM": 0.02, "IBRION": 6}, freq=True)
    text = path.read_text()
    assert "IBRION = 6" in text
    assert "POTIM = 0.02" in text
    assert "NSW = 1" in text  # safety default stays


def test_kpoints(tmp_path):
    path = write_kpoints(tmp_path, (6, 6, 1))
    assert "6 6 1" in path.read_text()


def _write_outcar(tmp_path, text):
    (tmp_path / "OUTCAR").write_text(text, encoding="utf-8")
    return tmp_path


def test_parse_energy_prefers_outcar_toten(tmp_path):
    _write_outcar(tmp_path,
                  "  free  energy   TOTEN        =     -548.111111 eV\n")
    assert parse_vasp_energy(tmp_path) == -548.111111


def test_parse_energy_falls_back_to_oszicar(tmp_path):
    (tmp_path / "OSZICAR").write_text("F=  -100.5\n", encoding="utf-8")
    assert parse_vasp_energy(tmp_path) == -100.5


def test_parse_converged_and_forces(tmp_path):
    _write_outcar(
        tmp_path,
        "  MAX TOTAL FORCE=  0.019\n"
        "  MAX TOTAL FORCE=  0.005\n"
        "  reached required accuracy - stopping structural energy minimisation\n")
    assert parse_vasp_converged(tmp_path)
    forces = parse_vasp_forces(tmp_path)
    assert forces == [0.019, 0.005]


def test_parse_frequencies_real_format(tmp_path):
    """Real VASP OUTCAR eigenvalue lines (f = ... cm-1) parse correctly."""
    _write_outcar(
        tmp_path,
        " Eigenvectors and eigenvalues of the dynamical matrix\n"
        " ----------------------------------------------------\n"
        "   1 f  =   63.924205 THz   401.645593 2PiTHz  2131.926900 cm-1   264.344497 meV\n"
        "   2 f  =   21.123456 THz   132.700000 2PiTHz   704.500000 cm-1    87.300000 meV\n"
        "   3 f/i=    1.100000 THz     6.900000 2PiTHz    36.700000 cm-1     4.500000 meV\n")
    parsed = parse_vasp_frequencies(tmp_path)
    assert parsed["has_freq_block"]
    assert parsed["frequencies_cm1"] == [2131.9269, 704.5, -36.7]
    assert parsed["n_imaginary"] == 1


def test_parse_frequencies_absent_block(tmp_path):
    _write_outcar(tmp_path, "  free  energy   TOTEN        =     -1.0 eV\n")
    parsed = parse_vasp_frequencies(tmp_path)
    assert not parsed["has_freq_block"]
    assert parsed["frequencies_cm1"] == []


def test_parse_frequencies_missing_outcar(tmp_path):
    parsed = parse_vasp_frequencies(tmp_path)
    assert not parsed["has_freq_block"]
    assert parsed["source"] == ""
