# Reproduce: Mask-aware dipole suite (RUN_MASKAWARE_SUITE_V1)

This file records the exact commands used to generate the main tables and figures.

## 1) Baseline SDSS + TNG comparison
python run_maskaware_compare.py \
  --sdss data/sdss_dr8/sdss_dr8_analysis_base_v1.csv \
  --tng50 data/tng50/tng50_sky_catalog_snap072.csv \
  --tng300 data/tng300/tng300_sky_catalog_snap072.csv \
  --n-null 2000 --seed 0 --zmin 0.0 --zmax 1.0

Outputs:
- results/maskaware_compare/maskaware_summary.csv
- results/maskaware_compare/maskaware_summary_table.tex
- results/maskaware_compare/fig_maskaware_sdss_tng_compare.png

## 2) Snap-specific z-cuts comparison
python run_maskaware_compare_zcuts.py \
  --sdss  data/sdss_dr8/sdss_dr8_analysis_base_v1.csv \
  --tng50 data/tng50/tng50_sky_catalog_snap072.csv \
  --tng300 data/tng300/tng300_sky_catalog_snap072.csv \
  --n-null 2000 --seed 0 \
  --sdss-zmin 0.02 --sdss-zmax 0.10 \
  --tng50-zmin 0.00 --tng50-zmax 0.01 \
  --tng300-zmin 0.00 --tng300-zmax 0.06

Outputs:
- results/maskaware_compare_zcuts/maskaware_summary.csv
- results/maskaware_compare_zcuts/maskaware_summary_table.tex
- results/maskaware_compare_zcuts/fig_maskaware_sdss_tng_compare.png

## 3) Robustness suite (hemisphere, jackknife, alt nulls, trimming)
python run_maskaware_robustness_suite.py \
  --input data/sdss_dr8/sdss_dr8_analysis_base_v1.csv \
  --label sdss_mass_z020_0100 \
  --col-ra RA --col-dec DEC --col-z Z --col-lgm LGM_TOT_P50 \
  --col-reliable RELIABLE --zmin 0.02 --zmax 0.10 \
  --weight-mode mass --n-null 2000 --seed 0 \
  --do-hemis --do-jackknife --jk-sectors 8 --jk-n-null 1000 \
  --do-alt-nulls --alt-n-null 2000 \
  --do-trim --trim-fracs 0,0.01,0.005,0.001 --trim-n-null 1000 \
  --progress

python run_maskaware_robustness_suite.py \
  --input data/sdss_dr8/sdss_dr8_analysis_base_v1.csv \
  --label sdss_rankmass_z020_0100 \
  --col-ra RA --col-dec DEC --col-z Z --col-lgm LGM_TOT_P50 \
  --col-reliable RELIABLE --zmin 0.02 --zmax 0.10 \
  --weight-mode rankmass --n-null 2000 --seed 0 \
  --do-hemis --do-jackknife --jk-sectors 8 --jk-n-null 1000 \
  --do-alt-nulls --alt-n-null 2000 \
  --do-trim --trim-fracs 0,0.01,0.005,0.001 --trim-n-null 1000

python run_maskaware_robustness_suite.py \
  --input data/tng300/tng300_sky_catalog_snap072_sdsscols.csv \
  --label tng300_mass_z000_0060 \
  --col-ra RA --col-dec DEC --col-z Z --col-lgm LGM_TOT_P50 \
  --col-reliable RELIABLE --zmin 0.00 --zmax 0.06 \
  --weight-mode mass --n-null 2000 --seed 0 \
  --do-hemis --do-jackknife --jk-sectors 8 --jk-n-null 1000 \
  --do-alt-nulls --alt-n-null 2000 \
  --do-trim --trim-fracs 0,0.01,0.005 --trim-n-null 1000

python run_maskaware_robustness_suite.py \
  --input data/tng50/tng50_sky_catalog_snap072_sdsscols.csv \
  --label tng50_mass_z000_0010 \
  --col-ra RA --col-dec DEC --col-z Z --col-lgm LGM_TOT_P50 \
  --col-reliable RELIABLE --zmin 0.00 --zmax 0.01 \
  --weight-mode mass --n-null 2000 --seed 0 \
  --do-hemis --do-jackknife --jk-sectors 8 --jk-n-null 1000 \
  --do-alt-nulls --alt-n-null 2000

## 4) Field rotation test (rotate mass field relative to footprint)
python run_maskaware_rotate_massfield.py \
  --input data/sdss_dr8/sdss_dr8_analysis_base_v1.csv \
  --label sdss_mass_rotatefield_z020_0100 \
  --col-ra RA --col-dec DEC --col-z Z --col-lgm LGM_TOT_P50 \
  --col-reliable RELIABLE --zmin 0.02 --zmax 0.10 \
  --weight-mode mass --n-null 2000 --n-rot 300 --seed 0 --make-fig

python run_maskaware_rotate_massfield.py \
  --input data/sdss_dr8/sdss_dr8_analysis_base_v1.csv \
  --label sdss_rankmass_rotatefield_z020_0100 \
  --col-ra RA --col-dec DEC --col-z Z --col-lgm LGM_TOT_P50 \
  --col-reliable RELIABLE --zmin 0.02 --zmax 0.10 \
  --weight-mode rankmass --n-null 2000 --n-rot 300 --seed 0 --make-fig

python run_maskaware_rotate_massfield.py \
  --input data/tng300/tng300_sky_catalog_snap072_sdsscols.csv \
  --label tng300_mass_rotatefield_z000_0060 \
  --col-ra RA --col-dec DEC --col-z Z --col-lgm LGM_TOT_P50 \
  --col-reliable RELIABLE --zmin 0.00 --zmax 0.06 \
  --weight-mode mass --n-null 2000 --n-rot 300 --seed 0 --make-fig

## 5) Weight-mode sensitivity (mass, logmass, clippedmass, rankmass)
python run_maskaware_weightmode_sensitivity.py \
  --label weightmodes_zcuts_v1 \
  --n-null 2000 --seed 0 --progress \
  --sdss  data/sdss_dr8/sdss_dr8_analysis_base_v1.csv \
  --tng50 data/tng50/tng50_sky_catalog_snap072_sdsscols.csv \
  --tng300 data/tng300/tng300_sky_catalog_snap072_sdsscols.csv \
  --sdss-zmin 0.02 --sdss-zmax 0.10 \
  --tng50-zmin 0.00 --tng50-zmax 0.01 \
  --tng300-zmin 0.00 --tng300-zmax 0.06

## 6) Random observer suite (if TNG HDF5 paths are correct)
python run_tng_random_observer_suite.py \
  --make-script make_tng_sky_catalog.py \
  --tng50-root /mnt/g/TNG50-1 \
  --tng300-root /mnt/g/TNG300-1 \
  --snap 72 \
  --label random_observer_suite_v1 \
  --n-observers 20 --seed0 0 \
  --mstar-min 1e9 \
  --n-null 2000 \
  --weight-modes mass,rankmass \
  --tng50-zmin 0.00 --tng50-zmax 0.01 \
  --tng300-zmin 0.00 --tng300-zmax 0.06
