# ccrcc is the acronym for Clear Cell Carcinoma
# and we will be using the cptac repo to get the omics.
# they have great documentation but we just need to get the gene,protein and clinical data from there.
# [https://github.com/PayneLab/cptac](https://github.com/PayneLab/cptac)  
import cptac
import pandas as pd
from pathlib import Path


# This will generate aninstance to the ccrcc omics data.
ccrcc_instance = cptac.Ccrcc()

#This will list all the data sources for ccrcc
ccrcc_instance.list_data_sources()
# breakpoint()

# Based on the output above we get a list of possible sources for ccrcc
# I did some digging already on what we would need from each one.


# each variable is alrady a pandas dataframe
genomics = ccrcc_instance.get_dataframe("CNV", "bcm")
proteomics =  ccrcc_instance.get_proteomics("bcm")

# this stands for Mount Sinai School of Medicine
clinical = ccrcc_instance.get_clinical("mssm")

print(f"patients in genes file: {len(genomics.index)}")
print(f"patients in proteins file: {len(proteomics.index)}")
print(f"patients in clinical data: {len(clinical.index)}")

#  data returned for our variables  is multi-indexed. Meaning it has multi-level headers
# The cptac repo has a utils class that makes it much easier to get the right index in place
# I fought for so long before finding this :(

# If what I said above is confusing, try uncommening the lines below to see data in csv format.
# we can save to a csv to explore further
root_dir = Path("Dataset")
genomics.to_csv(root_dir / "genes_raw.csv")
proteomics.to_csv(root_dir / "proteins_raw.csv")
clinical.to_csv(root_dir / "clinical_raw.csv")

proteomics = cptac.utils.reduce_multiindex(df=proteomics, levels_to_drop="Database_ID", quiet=True)
genomics = cptac.utils.reduce_multiindex(df=genomics, levels_to_drop="Database_ID", quiet=True)

# Lets try to get the same subjects/ patients
# Smalles dataset will drive this

print(f"Number of patients in the genes file: {genomics.index}")
print(f"Number of patients in proteins file: {proteomics.index}")
print(f"Number of patient in the clinical data: {clinical.index}")

# to get common patients we can take the intersection of each sets
# Assumming the first column is for patient ID

common_patients = clinical.index.intersection(genomics.index).intersection(proteomics.index)
print(f"Number of common patients: {len(common_patients)}")

# now we can use the list to subset and merge the datasets
# common_patients = list(common_patients)

genomics = genomics.loc[common_patients]
proteomics = proteomics.loc[common_patients]
clinical = clinical.loc[common_patients]

# to check
print("After filtering for common patients:")
print(f"Genomics: {genomics.shape} Proteomics: {proteomics.shape}  Clinical: {clinical.shape}")

# after the filtering we can save the to csv file again to see what the data looks like.
output_dir = Path("jessica_output")
output_dir.mkdir(exist_ok=True)
genomics.to_csv(output_dir / "genes_filtered.csv")
proteomics.to_csv(output_dir / "proteins_filtered.csv")
clinical.to_csv(output_dir / "clinical_filtered.csv")

# Up next! 
# we need to write a function to to encode the clincal data.
# for this we may want to look up which columns are the most relevant for ccrcc since there are too may features and mary are irrelevant.
# I feel like 6-10 features should be good. More is fine but need to be relevant to the cancer.