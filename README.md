# A Multimodal Approach For Enhanced Prediction in Renal Cell Carcinoma Detection

## Jessica Tan

## Vicente Ramos

clone the repo
create a virtual enviroment

`python -m venv .venv`

install dependencies

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

- Tools:

  - Pyradiomics: Open-source python package for the extraction of Radiomics features from 2D and 3D images and binary masks.

    - link : https://github.com/AIM-Harvard/pyradiomics
    - docs: https://pyradiomics.readthedocs.io/en/latest/
    - it support ct scan data based on what I saw on their docs. I did see this: `voxelArrayShift [0]: Integer, This amount is added to the gray level intensity in features Energy, Total Energy and RMS, this is to prevent negative values. If using CT data, or data normalized with mean 0, consider setting this parameter to a fixed value (e.g. 2000) that ensures non-negative numbers in the image. Bear in mind however, that the larger the value, the larger the volume confounding effect will be.`
    - tools seems a bit outdated. We need to test to see if it works ok

  - FAST: A framework for high-performance medical image processing, neural network inference and visualization
    - link: https://github.com/smistad/FAST/
    - docs: https://fast.eriksmistad.no/python-tutorial-mri-ct.html
    - this one seems a lot more extensive: a lot of funcitonality and neural based approaches.
    - It could be more challenging to start using due to its complexity and learning curve

- Data:
  - metadata.csv: contains information for each patient, study, images,etc.
  - Images: there is a total of 99,099 images for 877
    patients.
  - Format: Image files are in .dcm formart (DICOM)
  - CPTAC-CCRCC Folder
    - Size: 52.6GB
    - Number of folders: 877 folders corresponging to each patient
    - Each folder will have n number of CT images
    - Due to the size of the images we cannot push this to github. Please the images from the following link. https://www.cancerimagingarchive.net/collection/cptac-ccrcc/
    - To download the images you will need to install the data retrieving tool: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
- Image Data Workflow Overview:

Extracted relevant image features from CT scans for kidney cancer diagnosis using the two tools from above. The goal is to leverage these image features combined with clinical metadata to build predictive models.

- Files Created:

  - find_relevant.py:
    Splits the original CPTAC-CCRCC patient folders into two directories: present for patients with matching clinical data and not_present for the rest.

  - fast_extraction.py:
    Uses FAST to load DICOM series from the present directory. Saves the resulting images as NumPy arrays in npy_arrays and computes basic image statistics. Extracted features are stored in fast_features.csv.

  - pyradiomics.py:
    Loads the same NumPy arrays, converts them into SimpleITK images, and extracts a comprehensive set of radiomics features using PyRadiomics. Extracted features are stored in pyradiomics_features.csv.

  - model.py:
    Merges the image features from either FAST or PyRadiomics with the encoded clinical data. The clinical tumor stage is used as the label.

- Usage:

  - Run find_relevant.py to organize the patient directories.
  - Run fast_extraction.py or pyradiomics.py or both to extract and save image features along with the corresponding NumPy arrays.
  - Run model.py to merge image features with clinical data and train a predictive model. You can choose between FAST or PyRadiomics features for training.

Unique Patients clinical: 103
Moved 27 folders to /Users/jessicatan/CU Denver 2024/Spring 2025/CSCI 5930/Research Project/Clear-Cell-Carcinoma-Study/CT-Scan/present
Moved 38 folders to /Users/jessicatan/CU Denver 2024/Spring 2025/CSCI 5930/Research Project/Clear-Cell-Carcinoma-Study/CT-Scan/not_present
