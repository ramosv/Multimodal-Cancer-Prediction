import cptac
from pathlib import Path
from MultiModal_CNN import convert_to_png, check_blank_images, convert_single_dicom, predict_cnn_only, predict_prefusion, extract_all, run_predictions, freezing_cnn, find_overlap,encode_clinical_data,pre_process_omics, random_forest_classifier
import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_raw_data():
    # ccrcc is the acronym for Clear Cell Carcinoma
    # and we will be using the cptac repo to get the omics.
    # they have great documentation but we just need to get the gene,protein and clinical data from there.
    # [https://github.com/PayneLab/cptac](https://github.com/PayneLab/cptac)  
    ccrcc_instance = cptac.Ccrcc()

    #This will list all the data sources for ccrcc
    ccrcc_instance.list_data_sources()

    # Based on the output above we get a list of possible sources for ccrcc
    # I did some digging already on what we would need from each one.

    # each variable is alrady a pandas dataframe
    genomics = ccrcc_instance.get_dataframe("CNV", "bcm")
    proteomics =  ccrcc_instance.get_proteomics("bcm")

    # this stands for Mount Sinai School of Medicine
    clinical = ccrcc_instance.get_clinical("mssm")

    logging.info(f"patients in genes file: {len(genomics.index)}")
    logging.info(f"patients in proteins file: {len(proteomics.index)}")
    logging.info(f"patients in clinical data: {len(clinical.index)}")

    #  data returned for our variables  is multi-indexed. Meaning it has multi-level headers
    # The cptac repo has a utils class that makes it much easier to get the right index in place
    # I fought for so long before finding this :(

    # If what I said above is confusing, try uncommening the lines below to see data in csv format.
    # we can save to a csv to explore further
    # root_dir = Path("Dataset")
    # root_dir.mkdir(exist_ok=True)
    # genomics.to_csv(root_dir / "genes_raw.csv")
    # proteomics.to_csv(root_dir / "proteins_raw.csv")
    # clinical.to_csv(root_dir / "clinical_raw.csv")

    proteomics = cptac.utils.reduce_multiindex(df=proteomics, levels_to_drop="Name", quiet=True)
    genomics = cptac.utils.reduce_multiindex(df=genomics, levels_to_drop="Name", quiet=True)

    # Lets try to get the same subjects/ patients
    # Smalles dataset will drive this

    logging.info(f"Number of patients in the genes file: {genomics.index}")
    logging.info(f"Number of patients in proteins file: {proteomics.index}")
    logging.info(f"Number of patient in the clinical data: {clinical.index}")

    # to get common patients we can take the intersection of each sets
    # Assumming the first column is for patient ID

    common_patients = clinical.index.intersection(genomics.index).intersection(proteomics.index)
    logging.info(f"Number of common patients: {len(common_patients)}")

    # now we can use the list to subset and merge the datasets
    # common_patients = list(common_patients)

    genomics = genomics.loc[common_patients]
    proteomics = proteomics.loc[common_patients]
    clinical = clinical.loc[common_patients]

    # to check
    logging.info("After filtering for common patients:")
    logging.info(f"Genomics: {genomics.shape} Proteomics: {proteomics.shape}  Clinical: {clinical.shape}")

    # after the filtering we can save the to csv file again to see what the data looks like.
    output_dir = Path("all_output")
    output_dir.mkdir(exist_ok=True)
    genomics.to_csv(output_dir / "genes_filtered.csv")
    proteomics.to_csv(output_dir / "proteins_filtered.csv")
    clinical.to_csv(output_dir / "clinical_filtered.csv")

    # set the index to column 0
    # this is the patient ID column
    genomics_matched = pd.read_csv(output_dir / "genes_filtered.csv", index_col=0)
    proteomics_matched = pd.read_csv(output_dir / "proteins_filtered.csv", index_col=0)
    clinical_matched = pd.read_csv(output_dir / "clinical_filtered.csv",index_col=0)

    return genomics_matched, proteomics_matched, clinical_matched

if __name__ == "__main__":
    # Load the data

    genomics, proteomics, clinical = load_raw_data()
    import sys
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s",stream = sys.stdout, force = True)


    clinical_filtered, phenotype_mapped = encode_clinical_data(clinical)
    mask = phenotype_mapped.notna()
    phenotype_mapped = phenotype_mapped.loc[mask].astype(int)
    logging.info(f"Phenotype mapped: {phenotype_mapped.value_counts}")

    genomics_filtered, proteomics_filtered = pre_process_omics(genomics, proteomics)
    predictions, model = random_forest_classifier(genomics_filtered, proteomics_filtered, clinical_filtered, phenotype_mapped)

    root = Path.cwd()
    logging.info(f"Current working directory: {root}")
    images = root / "CT-Scan/CPTAC-CCRCC"

    # image index col should be 0
    find_overlap(clinical_filtered, images)

    convert_to_png(root / "CT-Scan/present")
    convert_to_png(root / "CT-Scan/not_present")

    phenotype2 = phenotype_mapped.copy()
    
    logging.info("Extracting features...")
    extract_all(root/"CT-Scan", "20250513")
    run_predictions()
    freezing_cnn(root/"CT-Scan", "20250513")

    predict_cnn_only(phenotype_mapped)
    predict_prefusion(genomics, proteomics, clinical, phenotype2)


