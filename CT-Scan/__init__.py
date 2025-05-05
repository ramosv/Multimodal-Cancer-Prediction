from .omics_features import load_data,encode_clinical_data,pre_process_omics

__all__ = ["load_data", "encode_clinical_data", "pre_process_omics"]

#     genomics, proteomics, clinical = load_data()
#     clinical, phenotype = encode_clinical_data(clinical)
#     genomics, proteomics = pre_process_omics(genomics,proteomics)