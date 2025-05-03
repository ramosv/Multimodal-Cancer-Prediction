import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from sklearn.model_selection import train_test_split
import logging
from omics_features import pre_process_omics, load_data, encode_clinical_data
# import PCA
from sklearn.decomposition import PCA



if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    # a quick test to ensure mps is responsive
    try:
        x = torch.ones(1, device=device)
        logging.info("Using MPS device.")
    except Exception as e:
        logging.warning(f"Error with MPS device: {e}. Falling back to CPU.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    logging.info("Falling back to CPU.")

class ClassifierNN(nn.Module):
    #input dim should be 128
    def __init__(self, input_dim, num_classes):
        super(ClassifierNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.fc(x)
    
def predict_prefusion(phenotype):
    root = Path("all_output")
    num_classes = 2
    features_dict = torch.load(root /"features.pt") 
    patient_ids = list(features_dict.keys())
    phenotype = phenotype.squeeze()  

    # get the features from the omics data#     
    genomics, proteomics, clinical,  = load_data()
    #clinical, phenotype = encode_clinical_data(clinical)
    genomics, proteomics = pre_process_omics(genomics,proteomics)

    #patient_ids = list(cnn_feateures.keys())
     
    # merge the features from omics + images in one feature vector
    omics = pd.concat([genomics, proteomics], axis=1, join="inner")
    omics_filtered = omics.loc[omics.index.isin(patient_ids)]
    omics_filtered = omics_filtered.reset_index(drop=True)

    # Create a DataFrame for CNN features without averaging:
    # For each patient, flatten the tensor [n, D] into a 1D vector of length n * D.
    cnn_flat_features = {}
    for pid, feat_data in features_dict.items():
        tensor = feat_data["features_avgpool"] 
        flat_feat = tensor.view(-1).cpu().numpy()  
        cnn_flat_features[pid] = flat_feat


    # Convert the dict of flattened features into a DataFrame
    cnn_df = pd.DataFrame.from_dict(cnn_flat_features, orient='index')
    cnn_df.index.name = "Patient_ID"
    
    # Merge CNN and omics features:
    # Find patients present in both datasets.
    common_patients = omics.index.intersection(cnn_df.index)
    omics_filtered = omics.loc[common_patients].copy()
    cnn_filtered = cnn_df.loc[common_patients].copy()
    
    cnn_filtered.dropna(axis=1, inplace=True)
    pca = PCA(n_components=256)
    cnn_reduced = pca.fit_transform(cnn_filtered)
    cnn_reduced_df = pd.DataFrame(cnn_reduced,
                                  index=cnn_filtered.index,
                                  columns=[f'pca_{i+1}' for i in range(256)])

    all_features = pd.concat([omics_filtered, cnn_reduced_df], axis=1)

    all_features.dropna(axis=1, inplace=True)

    phenotype_filtered = phenotype.loc[phenotype.index.intersection(common_patients)]

    print(all_features)
    print(phenotype_filtered)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    scaled_features_df = pd.DataFrame(scaled_features, index=all_features.index, columns=all_features.columns)
    
    print("Merged features shape:", scaled_features_df.shape)
    print(scaled_features_df.describe())
    
    # Train-test split (using RandomForestClassifier here instead of the NN)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features_df, phenotype_filtered, test_size=0.4, random_state=223, shuffle=True
    )
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))
    return predictions, model

def get_target_class(clinical):
    present_patients = Path("CT-Scan/present_png")
    relevant_patients =[]
    for folder in present_patients.iterdir():
        parts = folder.name.split("_")
        relevant_patients.append(parts[0])
    print(relevant_patients)
    clinical.reset_index(inplace=True)
    print(clinical.columns)
    clinical = clinical[clinical["Patient_ID"].isin(relevant_patients)]
    clinical = clinical[["Patient_ID", "tumor_stage_pathological"]]
    print(clinical)
    clinical.set_index("Patient_ID", inplace=True)
    print(clinical)


    # Construct target variable
    stages_dict = {
        "Stage I": 0,
        "Stage II": 1,
        "Stage III": 1,
        "Stage IV": 1
    }
    clinical["tumor_stage_pathological"] = clinical["tumor_stage_pathological"].map(stages_dict)
    phenotype = clinical[["tumor_stage_pathological"]]

    print(type(phenotype))
    print(phenotype)
    print(phenotype.value_counts(sort=True))
    return phenotype




def predict_cnn_only(phenotype):
    root = Path("all_output")
    num_classes = 2
    features_dict = torch.load(root /"custom_feats.pt") 
    labels_dict = torch.load(root /"labels.pt")
    patient_ids = list(labels_dict.keys())
    patients = []
    for key, value in labels_dict.items():
        if value == 1 and key in phenotype.index:
            patients.append(key)
    print("Patients:", patients)
    print("Number of patients:", len(patients))
    phenotype = phenotype.loc[phenotype.index.isin(patients)]
    print(phenotype.shape)
    phenotype = phenotype.squeeze()  
    rows = []
    for pid, feats in features_dict.items():
        if pid not in patients:
            continue
        mean_feat = feats.mean(dim=0).numpy()
        label = labels_dict[pid]
        row = {"Patient_ID": pid}
        for i in range(mean_feat.shape[0]):
            row[f"f{i}"] = mean_feat[i]
            print(f"f{i}: {mean_feat[i]}")
        
        row["label"] = label
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Patient_ID")
    
    X = df.drop(columns="label")
    y = df["label"]
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.long)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y, shuffle=True)

    # Initialize model
    input_dim = 128
    num_classes = 2

    model = ClassifierNN(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Evaluate
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy:.4f}")
    

    # # 4. Scale, split, train
    # scaler = StandardScaler()
    # Xs = scaler.fit_transform(X)

    # X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42, stratify=y)

    # # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # # clf.fit(X_train, y_train)
    # # pred = clf.predict(X_test)

    
    # # print("Accuracy:", accuracy_score(y_test, pred))
    # # # print(classification_report(y_test, pred))

    # X, y = [], []
    # for pid in patient_ids:
    #     if pid in phenotype.index:
    #         # 
    #         avg_features = features_dict[pid]["features_avgpool"].mean(dim=0)  
    #         X.append(avg_features.numpy())

    #         # If phenotype is truly a Series, this returns a single value:
    #         y.append(phenotype.loc[pid])

    # X = torch.tensor(X, dtype=torch.float32)  
    # y = torch.tensor(y, dtype=torch.long)

    # # Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=223, shuffle=True)

    # # Initialize model
    # input_dim = X_train.shape[1]  # Feature dimension from CNN
    # num_classes = 2  # Assuming binary classification: Stage 1 vs. Stage 2

    # model = ClassifierNN(input_dim, num_classes)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Training loop
    # for epoch in range(100):
    #     optimizer.zero_grad()
    #     outputs = model(X_train)
    #     loss = criterion(outputs, y_train)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # # Evaluate
    # with torch.no_grad():
    #     preds = model(X_test).argmax(dim=1)
    #     accuracy = (preds == y_test).float().mean().item()
    #     print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    genomics, proteomics, clinical = load_data()
    phenotype = get_target_class(clinical)
    phenotype2 = phenotype.copy()

    predict_cnn_only(phenotype)
    # predict_prefusion(phenotype2)
