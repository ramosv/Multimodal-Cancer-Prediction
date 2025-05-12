import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import logging

def load_data():
    root = Path("jessica_output")
    genomics = pd.read_csv(root / "genes_filtered.csv", index_col=0)
    proteomics = pd.read_csv(root / "proteins_filtered.csv", index_col=0)
    clinical = pd.read_csv(root / "clinical_filtered.csv",index_col=0)
    return genomics, proteomics, clinical

def encode_clinical_data(clinical):
    '''
    In this function we need to do a few things: 
    1. Based on the columns we selected during our zoom meeting, we need to clean the dataframe to only get those columns
    2. Construct our target variable using the stage cancer column
    3. Encode all string columns to numerical values
    '''
    
    relevant_columns = ["age", "sex", "tumor_laterality", "tumor_size_cm", "tumor_necrosis", "tumor_stage_pathological", "bmi", "alcohol_consumption", "tobacco_smoking_history", "medical_condition"]
    clinical = clinical[relevant_columns]

    # Construct target variable
    stages_dict = {
        "Stage I": 0,
        "Stage II": 1,
        "Stage III": 1,
        "Stage IV": 1
    }
    clinical["tumor_stage_pathological"] = clinical["tumor_stage_pathological"].map(stages_dict)
    phenotype = clinical["tumor_stage_pathological"]


    clinical = clinical.drop(columns="tumor_stage_pathological")
    clinical["age"].loc[clinical["age"] == ">=90"] = 90

    logging.info(type(phenotype))
    phenotype.squeeze()
    logging.info(phenotype)
    
    string_features = ["sex", "tumor_laterality", "tumor_necrosis","alcohol_consumption", "tobacco_smoking_history", "medical_condition"]
    encoder = LabelEncoder()
    clean_clinical = clinical.copy()

    for column in string_features:
        clean_clinical[column]  = encoder.fit_transform(clinical[column])

    logging.info(phenotype.value_counts(sort=True))
    logging.info(clean_clinical)
    
    #alcohol_consumption = [""]
    # age, sex, tumor_laterality, tumor_size_cm, tumor_necrosis, tumor_stage_pathological, bmi,
    # alcohol_consumption, tobacco_smoking_history, medica_condition, follow_up_period, vital_status_at_date_of_last_contact,
    # tumor_status_at_date_of_last_contact_or_death, Survival status (1, dead; 0, alive)

    # Based on the remaining columns we will need to encode them.
    # Pandas has a function called get_dummies that can help us lol
    #clinical = pd.get_dummies(clinical)

    # at the end we return the a tuple with the clean clinica df and the phenotype or target variable
    return clean_clinical, phenotype

def pre_process_omics(genomics, proteomics):
    # Genomics and proteomics are likley to have irrelant columns that we need to drop
    # They are also likely to have missing values that we need to fill and columns with huge variance that we need to scale
    # we cna plot some of these to see the distribution of the data
    # we can also use the describe function to get a summary of the data
    # we can also use the corr function to see the correlation between the columns
    # at the end we can return a cleaned out version of each dataset

    genomics = genomics.fillna(0)
    proteomics = proteomics.fillna(0)
    # df = df.loc[:, (df==0).mean() < .7]
    logging.info("Before filtering:")
    logging.info(genomics.shape)
    logging.info(proteomics.shape)
    logging.info(genomics.describe())
    logging.info(proteomics.describe())

    # remove columns with more than 50% missing values
    genomics = genomics.loc[:, (genomics == 0).mean() <= 0.5]
    proteomics = proteomics.loc[:, (proteomics == 0).mean() <= 0.5]

    # Drop features with very low variance
    from sklearn.feature_selection import VarianceThreshold
    genomics_index = genomics.index
    genomics_columns = genomics.columns
    proteomics_index = proteomics.index
    proteomics_columns = proteomics.columns

    variance_filter = VarianceThreshold(threshold=0.05)
    genomics = variance_filter.fit_transform(genomics)
    genomics = pd.DataFrame(genomics, index=genomics_index, columns=genomics_columns[variance_filter.get_support()])
    proteomics = variance_filter.fit_transform(proteomics)
    proteomics = pd.DataFrame(proteomics, index=proteomics_index, columns=proteomics_columns[variance_filter.get_support()])

    logging.info("\nAfter filtering:")
    logging.info(genomics.shape)
    logging.info(proteomics.shape)

    # from numpy array to pandas dataframe
    logging.info(genomics.head())
    logging.info(proteomics.head())
    #logging.info(genomics.describe())
    #logging.info(proteomics.describe())

    return genomics, proteomics

def random_forest_classifier(genomics, proteomics, clinical, phenotype):
    '''
    Here we will train the model and return the predictions and metrics 
    '''
    logging.info('Starting random forest classifier')
    # merge the omics
    logging.info(genomics.shape)
    logging.info(proteomics.shape)
    logging.info(clinical.shape)
    omics = pd.concat([genomics, proteomics, clinical], axis=1, join="inner")

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(omics, phenotype, test_size=0.3, random_state=127)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    logging.info(classification_report(y_test, predictions))
    logging.info(accuracy_score(y_test, predictions))
    return predictions, model

# if __name__ == "__main__":
#     genomics, proteomics, clinical = load_data()
#     clinical, phenotype = encode_clinical_data(clinical)
#     genomics, proteomics = pre_process_omics(genomics,proteomics)
#     predictions, model = random_forest_classifier(genomics, proteomics, clinical, phenotype)

