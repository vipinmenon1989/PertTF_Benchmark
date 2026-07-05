# Execution Log

Pipeline started: Saturday, Jul 4, 2026

## Phase 0: Logging Protocol
- Created `execution_log.md` and `error_log.md` in repository root.

## Phase 1: Code Review & Error Resolution
- Scanned all files in repository (5 Python scripts, h5ad data, result directories).
- **Fixed** `LGB_genotype.py`: output directory was `./random_split` but results live in `genotype_split/`; renamed to `GENOTYPE_DIR = "./genotype_split"`.
- **Fixed** `scVI/scVI_analysis_celltype.py` and `scVI/scVI_analysis_genotype.py`: added `Path`-based resolution for h5ad input and output directories so scripts work from any working directory; removed unused `train_test_split` import.
- Syntax check (`python3 -m py_compile`) passed for all Python files.

## Phase 2: Environment & Snakemake Configuration
- Verified `pertTF_bench` conda environment exists at `/local/projects-t3/lilab/vmenon/anaconda3/envs/pertTF_bench`.
- Installed `snakemake` via `conda install -c bioconda -c conda-forge snakemake` (v9.22.0).
- Created `Snakefile` orchestrating: `pre_analysis` → parallel `lgb_celltype`, `lgb_genotype`, `scanvi_celltype`, `scanvi_genotype` → `rule all`.
- Dry run `snakemake -n` completed successfully (exit 0): DAG built, all target outputs present.

## Phase 3: CI/CD & Documentation
- Created `run_pipeline.slurm` with ihc account/partition, ihc-h200-1 node, 400G RAM, 600 min, 1 GPU.
- Created `.github/workflows/snakemake_ci.yml` (flake8 lint, py_compile, snakemake dry run).
- Created comprehensive `README.md` with purpose, script details, DAG diagram, and run instructions.
- Created `.gitignore` with strict exclusion policy per specification.
