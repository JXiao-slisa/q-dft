# Adsorption-energy workflow report (adsorption-full)

- generated: 2026-09-10T21:17:04.122333+00:00
- engine mode: **mock**  ⚠ SYNTHETIC mock data (Rule Zero)
- protocol: build -> MLIP relax -> DFT single point (three systems)
- elapsed: 1.1 s

## Inputs

```json
{
  "element": "Pt",
  "miller": "(111)",
  "layers": 3,
  "adsorbate": "OH",
  "site": "fcc",
  "dft_calculator": "vasp",
  "size": "2,2",
  "vacuum": 10.0,
  "fmax": 0.05,
  "mlip_steps": 200,
  "temperature": 298.15
}
```

## Single-point energies (eV)

| system | DFT energy |
|---|---|
| slab | 2.508708 |
| adsorbate | 2.179459 |
| slab_ads | 4.688215 |

**E_ad = E(slab+ads) - E(slab) - E(adsorbate) = 0.0 eV**

## Gibbs correction (adsorbate, empirical)

- G(adsorbate) = 1.814 eV
- E_ad(Gibbs approx.) = 0.3655 eV
- note: approximate: adsorbate empirical corrections; for the adsorbed state use vibrational_thermochemistry_tool

## Per-system details

### slab
- structure: `D:\Zcode\slisa\editions\q-dft\Pt_111_3l.vasp`
- MLIP optimized: `D:\Zcode\slisa\editions\q-dft\Pt_111_3l_slab_mlip_opt.vasp` (E=2.508707964720619, converged=True)
- DFT workdir: `slisa_vasp_mock_nboohrys\vasp` (converged=True)

### adsorbate
- structure: `D:\Zcode\slisa\editions\q-dft\OH.vasp`
- MLIP optimized: `D:\Zcode\slisa\editions\q-dft\OH_adsorbate_mlip_opt.vasp` (E=2.1794586741746698, converged=True)
- DFT workdir: `slisa_vasp_mock_60z8dk8y\vasp` (converged=True)

### slab_ads
- structure: `D:\Zcode\slisa\editions\q-dft\Pt_111_3l_OH_fcc.vasp`
- MLIP optimized: `D:\Zcode\slisa\editions\q-dft\Pt_111_3l_OH_fcc_slab_ads_mlip_opt.vasp` (E=4.688215263942662, converged=True)
- DFT workdir: `slisa_vasp_mock_1hli9s4t\vasp` (converged=True)

*All energies traceable to the calculation files in the referenced workdirs (Rule Zero).*