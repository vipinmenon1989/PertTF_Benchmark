"""
PertTF Benchmark Snakemake Workflow
Orchestrates LightGBM and scANVI classification benchmarks on single-cell data.
"""

import os

WORKDIR = workflow.basedir
H5AD = os.path.join(
    WORKDIR,
    "object_integrated_assay3_annotated_final.modified.cleaned.updated.celltype2_only.h5ad",
)


rule all:
    input:
        "celltype_split/overall_evaluation_metrics.csv",
        "genotype_split/overall_evaluation_metrics.csv",
        "scVI/scanvi_celltype_split/overall_evaluation_metrics.csv",
        "scVI/scanvi_genotype_split/overall_evaluation_metrics.csv",


rule pre_analysis:
    input:
        h5ad=H5AD,
    output:
        touch("pre_analysis.done"),
    log:
        "pre_analysis.log",
    shell:
        "python pre_analysis.py > {log} 2>&1"


rule lgb_celltype:
    input:
        h5ad=H5AD,
        pre="pre_analysis.done",
    output:
        "celltype_split/overall_evaluation_metrics.csv",
    log:
        "celltype_split/lgb_celltype.log",
    shell:
        "python LGB_celltype.py > {log} 2>&1"


rule lgb_genotype:
    input:
        h5ad=H5AD,
        pre="pre_analysis.done",
    output:
        "genotype_split/overall_evaluation_metrics.csv",
    log:
        "genotype_split/lgb_genotype.log",
    shell:
        "python LGB_genotype.py > {log} 2>&1"


rule scanvi_celltype:
    input:
        h5ad=H5AD,
        pre="pre_analysis.done",
    output:
        "scVI/scanvi_celltype_split/overall_evaluation_metrics.csv",
    log:
        "scVI/scanvi_celltype.log",
    resources:
        gpu=1,
    shell:
        "python scVI/scVI_analysis_celltype.py > {log} 2>&1"


rule scanvi_genotype:
    input:
        h5ad=H5AD,
        pre="pre_analysis.done",
    output:
        "scVI/scanvi_genotype_split/overall_evaluation_metrics.csv",
    log:
        "scVI/scanvi_genotype.log",
    resources:
        gpu=1,
    shell:
        "python scVI/scVI_analysis_genotype.py > {log} 2>&1"
