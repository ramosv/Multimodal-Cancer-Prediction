# A Multimodal Approach For Enhanced Prediction in Renal Cell Carcinoma Detection

Authors:
### Vicente Ramos
### Jessica Tan

## Setup Instructions:

0. Clone the repo:

  - `git clone git@github.com:ramosv/Multimodal-Cancer-Prediction.git`

1. Create a virtual environment:

  - `python -m venv .venv`

2. Activate the virtual environment:

  - Unix: `source .venv/bin/activate`
  - Windows: `source .venv/Scripts/activate`

3. Install dependencies:

  - `pip install -r requirements.txt`

  - Note: Install `torch` and `torchvision` from [pytorch.org](https://pytorch.org/get-started/locally/) according to your system requirements.

4. Run the pipeline:

  - `python main.py`

  - Note: Continue reading to see where to download the image data from.

## Datasets

### Omics & clinical dataset
  - Proteomics, Genomics and Clinical data was available through the `cptac` python package, more information available at: [github.com/PayneLab/cptac](https://github.com/PayneLab/cptac)

### Image data
All image data is publicly available from the [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cptac-ccrcc/)

Files:
  - metadata.csv: contains information for each patient, study, images,etc.
  - Images: there are `63` subjects with a total of `99,099` across all of them.
  - Format: Image files are in .dcm formart (DICOM)
  - CPTAC-CCRCC Folder (download required)
    - Size: 52.6GB
    - Number of folders: `63` folders corresponding to each patient. 
    - Number of sub-folders: `877` sub-folders.
    - Number of images: It varies per patient, for example: 
      - Subject `C3L-00815` has `2294` images.
      - Subject `C3N-00246` has `160` images.

    - Due to the large size of the images, we cannot push this data to the repository. Please download the images from the following link: [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cptac-ccrcc/)

    - To download the images you will need to git install a data retrieving tool. [NBIA](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images).

## Feature extraction

  - `find_relevant.py`: Splits the folders into two directories: `present` and `not_present`.
  - `dicom_to_png.py`: Converts `.dcm` images to `.png` format.
  - `cnn_extraction.py`: Extracts features from both `present` and `not_present` directories.
  - `cnn_prediction.py`: Uses the learned features to train a separate CNN model to predict cancer stage.

  - Intersecting genomic, proteomic, and clinical data yields 103 overlapping subjects.
  - Intersecting those 103 subjects with the 63 from the image data yields 25 overlapping subjects.
  - We still use the remaining 38 subjects to extract features. We apply a method called transfer learning, where a label is set up to guide the CNN model in feature extraction. In this project, we use a binary label to classify `present` vs. `not_present`. After training, we freeze the model weights and use the output from the penultimate layer. This output is then used to train a shallow model to predict cancer stage (our original label).

## Sources and Background:

  - Getting started: https://forum.image.sc/t/basics-of-ct-preprocessing/88549
  - Deep learning for end-to-end kidney cancer diagnosis on multi-phase abdominal computed tomography: https://www.nature.com/articles/s41698-021-00195-y
  - MSS U-Net: 3D segmentation of kidneys and tumors from CT images with a multi-scale supervised U-Net: https://www.sciencedirect.com/science/article/pii/S2352914820301969
  - Kidney Cancer Diagnosis and Surgery Selection by Machine Learning from CT Scans Combined with Clinical Metadata: https://pmc.ncbi.nlm.nih.gov/articles/PMC10296307/
  - Kidney Tumor Detection and Classification Based on Deep Learning Approaches: A New Dataset in CT Scans: https://onlinelibrary.wiley.com/doi/10.1155/2022/3861161
  - An Effective Method for Lung Cancer Diagnosis from CT Scan Using Deep Learning-Based Support Vector Network: https://pmc.ncbi.nlm.nih.gov/articles/PMC9657078/
  - Image processing techniques for analyzing CT scan images towards the early detection of lung cancer: https://pmc.ncbi.nlm.nih.gov/articles/PMC6822523/

