from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_clinical_data():
    root = Path("jessica_output")
    clinical = pd.read_csv(root / "clinical_filtered.csv", index_col=0)
    clinical["PatientID"] = clinical.index

    relevant_columns = [
        "age", "sex", "tumor_laterality", "tumor_size_cm", 
        "tumor_necrosis", "tumor_stage_pathological", "bmi", 
        "alcohol_consumption", "tobacco_smoking_history", "medical_condition", 
        "PatientID"
    ]
    clinical = clinical[relevant_columns]

    stages_dict = {
        "Stage I": 0,
        "Stage II": 1,
        "Stage III": 2,
        "Stage IV": 3
    }
    clinical["tumor_stage_pathological"] = clinical["tumor_stage_pathological"].map(stages_dict)
    phenotype = clinical["tumor_stage_pathological"]

    clean_clinical = clinical.drop(columns=["tumor_stage_pathological"])
    age_map = {">=90": 90}
    clean_clinical["age"] = clean_clinical["age"].replace(age_map)
    string_features = ["sex", "tumor_laterality", "tumor_necrosis",
                       "alcohol_consumption", "tobacco_smoking_history", "medical_condition"]
    
    encoder = LabelEncoder()
    for col in string_features:
        clean_clinical[col] = encoder.fit_transform(clean_clinical[col].astype(str))
    
    return clean_clinical, phenotype

def merge_with_clinical(image_features_csv):
    root = Path.cwd()
    features_df = pd.read_csv(root / image_features_csv)
    
    clinical, phenotype = load_clinical_data()
    
    merged = pd.merge(features_df, clinical, on="PatientID", how="inner")

    merged["Label"] = merged["PatientID"].map(phenotype)
    return merged

def train_and_evaluate(merged_df, label_column="Label"):
    drop_cols = ["PatientID", "Series", "Filename", label_column]
    feature_cols = [col for col in merged_df.columns if col not in drop_cols]
    X = merged_df[feature_cols]
    y = merged_df[label_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=127
    )
    model = RandomForestClassifier(random_state=127)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))
    return model, predictions

def test_fast_clinical():
    print("Training with FAST image features + clinical")
    merged = merge_with_clinical("fast_features.csv")
    train_and_evaluate(merged)

def test_radiomics_clinical():
    print("Training with PyRadiomics image features + clinical")
    merged = merge_with_clinical("pyradiomics_features.csv")
    train_and_evaluate(merged)

if __name__ == "__main__":
    test_fast_clinical()
    print("\n" + "="*50 + "\n")
    test_radiomics_clinical()
