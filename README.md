# A Multimodal Approach For Enhanced Prediction in Renal Cell Carcinoma Detection

### Jessica Tan
### Vicente Ramos

## setup instructions
clone the repo
1. create a virtual enviroment

`python -m venv .venv`

2. install dependencies

`pip install -r requirements.txt`

Starting work on the second modality (Image Data): Independent approach
Extracting relevant features from CT scan

- Background:

  - INTRO: this forum above has a good overview of the basics of ct preprocessing:https://forum.image.sc/t/basics-of-ct-preprocessing/88549
  - Deep learning for end-to-end kidney cancer diagnosis on multi-phase abdominal computed tomography: https://www.nature.com/articles/s41698-021-00195-y
  - MSS U-Net: 3D segmentation of kidneys and tumors from CT images with a multi-scale supervised U-Net: https://www.sciencedirect.com/science/article/pii/S2352914820301969
  - Kidney Cancer Diagnosis and Surgery Selection by Machine Learning from CT Scans Combined with Clinical Metadata: https://pmc.ncbi.nlm.nih.gov/articles/PMC10296307/
  - Kidney Tumor Detection and Classification Based on Deep Learning Approaches: A New Dataset in CT Scans: https://onlinelibrary.wiley.com/doi/10.1155/2022/3861161
  - An Effective Method for Lung Cancer Diagnosis from CT Scan Using Deep Learning-Based Support Vector Network: https://pmc.ncbi.nlm.nih.gov/articles/PMC9657078/
  - Image processing techniques for analyzing CT scan images towards the early detection of lung cancer: https://pmc.ncbi.nlm.nih.gov/articles/PMC6822523/


## Datasets

### Omics & clinical dataset
  - Proteomics, Genomics and Clinical data was pulled from the `cptac` python package, more information available at: [github.com/PayneLab/cptac](https://github.com/PayneLab/cptac)

### Image data
  - Data is publicly available from the [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cptac-ccrcc/)
  Files:
    - metadata.csv: contains information for each patient, study, images,etc.
    - Images: there is a total of `99,099` images for `877`
      patients.
    - Format: Image files are in .dcm formart (DICOM)
    - CPTAC-CCRCC Folder
      - Size: 52.6GB
      - Number of folders: `877` folders corresponging to each patient
      - Each folder will have `n` number of CT images
      - Due to the size of the images we cannot push this to github. Please the images from the following link. https://www.cancerimagingarchive.net/collection/cptac-ccrcc/
      - To download the images you will need to install the data retrieving tool: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images

- Image Data Workflow Overview:

Extracted relevant image features from CT scans for kidney cancer diagnosis using a Convolutional Neural Network (CNN)

### Python files:

  - find_relevant.py:
    Splits the original CPTAC-CCRCC patient folders into two directories: `present` for patients with matching clinical data and `not_present` for the rest.

  - 

- Usage:

  - Run find_relevant.py to organize the patient directories.
  - Run fast_extraction.py or pyradiomics.py or both to extract and save image features along with the corresponding NumPy arrays.

Unique Patients clinical: 103 (from cpta-ccrcc)
Moved 27 overlapping patients
Moved 38 folders to /Users/jessicatan/CU Denver 2024/Spring 2025/CSCI 5930/Research Project/Clear-Cell-Carcinoma-Study/CT-Scan/not_present


-First run find_relevant.py
-Run only_ct.py
/home/vicente/Github/Clear-Cell-Carcinoma-Study/CT-Scan/20250411/C3L-00800_png/20081130_125502_MR__20081130_125502_MR_
/home/vicente/Github/Clear-Cell-Carcinoma-Study/CT-Scan/20250411/C3L-00817_png/20080516_070458_MR__20080516_070458_MR_
/home/vicente/Github/Clear-Cell-Carcinoma-Study/CT-Scan/20250411/C3N-01808_png/20100601_083722_MR_6526676604424618_20100601_083722_MR_6526676604424618

pre fusion:
    Direct Concatenation with Alignment and Normalization

post fusion:

    Ensemble of Modal-Specific Models with Meta-Learning
    Cascade Decision Fusion
