# Analysis pipeline — run order and environment

## Environment

- Python 3.11 (packages in `requirements.txt`)
- Java (JDK 8+) on the JVM classpath for the JIDT library:
  the TE estimator loads `infodynamics.jar` (JIDT 1.5,
  https://github.com/jlizier/jidt). In the scripts the jar path is set by the
  constant `JAR` (edit to your local path).
- The manuscript figures were produced with the scripts in `figures_src/`
  (cartopy is only needed for the station-map figure).

## Run order

| Step | Script | Output | Manuscript item |
|---|---|---|---|
| 0 | `code/e0_data_prep.py` | `data/{STN}_2015_1min.csv.gz` | §2 data |
| 0b | `code/e0b_data_prep_dat.py` | variant input (SuperMAG `dat` sources) | not used in final manuscript |
| 1 | `code/e1_validate.py` | `results/e1_validation.csv` | Fig. 3 |
| 2 | `code/e2_te_scan.py` | `results/e2/` | Fig. 4 |
| 3 | `code/e3_significance.py` | `results/e3/` | Fig. 5, §4.2 |
| 4 | `code/e4_storm_quiet.py` | `results/e4/` | Fig. 9 |
| 5 | `code/e5_latitude_analysis.py` | `results/e3/e5_station_mlat.csv` | Fig. 10 |
| 6a | `code/e6a_history_scan.py` | `results/e6a/` | Fig. 6 |
| 6b | `code/e6b_robustness.py` | `results/e6b/` | Fig. 7 |
| 6c | `code/e6c_conditional_te.py` | `results/e6c/` | Fig. 8 |
| 6d | `code/e6d_perchunk.py` (+ `e6d_infomeasure.py`) | `results/e6d/` | Table 3 |
| 8 | `code/e8b_stratified.py` | `results/e8b/` | §4 auxiliary |
| 9 | `code/e9_cascade.py` | `results/e9/` | §4 auxiliary |
| 10 | `code/e10_controls.py`, `code/e10b_ulfindex.py` | `results/e10/` | §4.9, Fig. 11 |
| F | `figures_src/*.py` | manuscript figures | Figs. 1–11 |

Notes:

- `e2/e3` use the **destination history l = 1**; `e4/e6b/e6c/e6d` use l = 6
  (see manuscript §3.1/§4.3 for the embedding-choice justification). The
  history sweep (`e6a`) covers l = 1–24 and shows the excess TE is flat over
  this range for the pressure channel.
- The 1 Hz input records and the raw OMNI files are not redistributed; see
  README.md §3 for their public sources. `e0_data_prep.py` expects them in
  the paths configured at the top of the script.
- `e8_rbsp_prep.py` / `e8b_ephem_test.py` / `e0b_data_prep_dat.py` pertain to
  auxiliary explorations not included in the final manuscript.
