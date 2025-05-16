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
- Note: `dicom_to_png.py` file will create a folder timestamped with today's date. You will have to update it manually to the corresponding name to collect the correct images. In our case, it was '20250513'
- Note: Continue reading to see where to download the image data from.

## Datasets

### Multi-Omics Data:

Using CPTAC github repository is a python package that facilitates access to cancer data from the National Cancer institute.

- Genomics: 103 x 60,525
- Proteomics: 103 x 11,259
- Clinical: 103 x 124
  We used a variance threshold of 0.05 to reduce the number of features for genomics and proteomics. For clinical features, we selected the following columns based on our literature review.
- Genomics: 103 x 11,000
- Proteomics: 103 x 6930
- Clinical: ["age", "sex", "tumor_laterality", "tumor_size_cm", "tumor_necrosis", "tumor_stage_pathological", "bmi", "alcohol_consumption", "tobacco_smoking_history", "medical_condition"]

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

## Sample of Results

Using `cnn_extraction.py`, we created two baselines shown below.

Table 1: Binary Classifer Based on CPTAC-CCRCC Multi-Omics Dataset.

| Metric  | Precision | Recall | F1-Score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Class 0 | 0.53      | 0.60   | 0.56     | 15      |
| Class 1 | 0.57      | 0.59   | 0.53     | 16      |

**Overall Accuracy:** 0.55 (on 31 samples)
**Macro Average:** Precision = 0.55, Recall = 0.55, F1-Score = 0.55
**Weighted Average:** Precision = 0.55, Recall = 0.55, F1-Score = 0.55

Table 2: Binary Classifier based on CPTAC-CCRCC CT-Scan Dataset
| Class | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **0** | 0.55 | 0.73 | 0.63 | 15 |
| **1** | 0.64 | 0.44 | 0.52 | 16 |
| | | | | |
| **Accuracy** | | | 0.58 | 31 |
| **Macro Avg** | 0.59 | 0.59 | 0.57 | 31 |
| **Weighted Avg** | 0.59 | 0.58 | 0.57 | 31 |

Below are the results summary after filtering patients across both datasets

Table 3: CNN - Simple Transforms
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** | 0.50 | 0.58 | 0.54 | 12 |
| **1** | 0.17 | 0.12 | 0.14 | 8 |
| **Accuracy** 0.4000 |

Table 4: CNN - Added Image Augmentation
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** | 0.54 | 0.64 | 0.58 | 11 |
| **1** | 0.33 | 0.25 | 0.29 | 8 |
| **Accuracy** 0.4737 |

Results from CNN with cross-validation: Model trained over 2 folds with shallow architecture. Accuracy improves modestly over epochs.

- Fold 0: Final accuracy = 0.5000
- Fold 1: Final accuracy = 0.5385
- Mean accuracy across folds: 0.5192

Table 4: Pre-fusion (CNN + Omics) - Binary with Cross-Validation
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0** | 0.83 | 1.00 | 0.91 | 5 |
| **1** | 1.00 | 0.75 | 0.86 | 4 |
| **Accuracy** 0.8889 |

## Plots of Results

![Omics Baseline](/plots/output3.png)
![Precision](/plots/output.png)
![Recall](/plots/output1.png)
![f1](/plots/output2.png)

## Conclusion:

The potential of integrating CT imaging and multi-omics profiles for more accurate detection of renal cell carcinoma is demonstrated by our results. By developing separate baselines and combining them through a pre-fusion approach, we observed a meaningful improvement in classification performance, achieving an accuracy of up to 88.9% using cross-validation. These results suggest that leveraging complementary data sources can provide more robust insights than single-modality models alone. Future work may explore more post fusion strategies, larger sample sizes, and expanded label granularity to further enhance clinical relevance. We're also exploring testing our methods with other data sources so we can validate our results across different cancer types.

## Sources and Background:

- Getting started: https://forum.image.sc/t/basics-of-ct-preprocessing/88549
- Deep learning for end-to-end kidney cancer diagnosis on multi-phase abdominal computed tomography: https://www.nature.com/articles/s41698-021-00195-y
- MSS U-Net: 3D segmentation of kidneys and tumors from CT images with a multi-scale supervised U-Net: https://www.sciencedirect.com/science/article/pii/S2352914820301969
- Kidney Cancer Diagnosis and Surgery Selection by Machine Learning from CT Scans Combined with Clinical Metadata: https://pmc.ncbi.nlm.nih.gov/articles/PMC10296307/
- Kidney Tumor Detection and Classification Based on Deep Learning Approaches: A New Dataset in CT Scans: https://onlinelibrary.wiley.com/doi/10.1155/2022/3861161
- An Effective Method for Lung Cancer Diagnosis from CT Scan Using Deep Learning-Based Support Vector Network: https://pmc.ncbi.nlm.nih.gov/articles/PMC9657078/
- Image processing techniques for analyzing CT scan images towards the early detection of lung cancer: https://pmc.ncbi.nlm.nih.gov/articles/PMC6822523/
