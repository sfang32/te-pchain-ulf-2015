# Transfer entropy from solar wind to ground ULF Pc5 wave power: complete analysis software (JASTP-D-26-00601)

This is the **software deposit** accompanying the manuscript *"Transfer entropy from solar wind to ground ULF Pc5 wave power across the Chinese Meridian chain"*. The **companion data deposit** (derived 1-min band-power series for the 13 chain stations and all pipeline result tables) is archived on ScienceDB: **CSTR: 31253.11.sciencedb.0183j, DOI: https://doi.org/10.57760/sciencedb.0183j**.

## Contents

- `code/` — the complete analysis pipeline:  - `e0_data_prep.py`, `e0b_data_prep_dat.py` — data preparation (1 Hz → 1-min band power + OMNI merge)
  - `e1_validate.py` — estimator validation against analytic nulls
  - `e2_te_scan.py`, `e3_significance.py` — TE–lag scan and circular-shift surrogate significance (main result)
  - `e4_storm_quiet.py`, `e5_latitude_analysis.py` — stratified and latitude analyses
  - `e6a_history_scan.py`, `e6b_robustness.py`, `e6c_conditional_te.py`, `e6d_perchunk.py`, `e6d_infomeasure.py` — history sweep, robustness, conditional TE, block bootstrap
  - `e8_rbsp_prep.py`, `e8b_stratified.py`, `e8b_ephem_test.py`, `e9_cascade.py` — auxiliary analyses
  - `e10_controls.py`, `e10b_ulfindex.py` — revision control analyses (chain-average, local-time split, AE-conditioned, variability, ULF-index comparison)
  - `make_paper_figures.py` + `figures_src/` — figure generation
  - `smoke_test.py` — end-to-end smoke test
- `docs/` — `README_pipeline.md` (environment and run order)

## Environment

Python 3.11 with `requirements.txt`; TE estimation uses the JIDT library (`infodynamics.jar` 1.5, https://github.com/jlizier/jidt) through `jpype` and requires Java. Set the `JAR` constant in the scripts to your local jar path.

## Notes

- The raw 1 Hz magnetometer records and the raw OMNI files are not redistributed; see the companion data deposit and `e0_data_prep.py` for their public sources and expected paths.
- `e2/e3` use destination history l = 1; `e4/e6b/e6c/e6d` use l = 6 (justification in manuscript §3.1/§4.3; the sweep in `e6a` shows the pressure excess is flat for l = 1–24).
- `e8*`/`e0b` pertain to auxiliary explorations not included in the final manuscript.

## Licence

MIT (see `LICENSE`). Please cite this software and the companion data deposit (DOI: 10.57760/sciencedb.0183j); see `CITATION.cff`.
