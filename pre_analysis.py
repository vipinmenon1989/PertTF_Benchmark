import scanpy as sc

# 1. Read the h5ad file
# Backed mode 'r' is useful for 170k cells to save RAM if you just want to inspect metadata
adata = sc.read_h5ad("object_integrated_assay3_annotated_final.modified.cleaned.updated.celltype2_only.h5ad")
print(adata)

# 2. View all available metadata columns
print("\n--- Metadata Columns Available ---")
print(adata.obs.columns.tolist())

# 3. Inspect the first few rows of the metadata dataframe
print("\n--- First 5 Rows of Metadata ---")
print(adata.obs[["genotype", "celltype_2"]].head())

# 4. Check for Class Imbalance (Crucial for LGBM)
print("\n--- Genotype Distribution ---")
print(adata.obs["genotype"].value_counts())

print("\n--- Cell Type Distribution ---")
print(adata.obs["celltype_2"].value_counts())

# 5. Check for missing values in your target variables
print("\n--- Missing Values Check ---")
print(adata.obs[["genotype", "celltype_2"]].isnull().sum())
