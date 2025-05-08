import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.model_selection import train_test_split
import logging
from omics_features import pre_process_omics, load_data, encode_clinical_data
# import PCA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold


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

    
def predict_prefusion(phenotype):
    root = Path("all_output")
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
            #print(f"f{i}: {mean_feat[i]}")
        
        row["label"] = label
        rows.append(row)

    cnn_features = pd.DataFrame(rows).set_index("Patient_ID")
    print(cnn_features)

    phenotype = phenotype.squeeze()  

    # get the features from the omics data#     
    genomics, proteomics, clinical,  = load_data()
    #clinical, phenotype = encode_clinical_data(clinical)
    genomics, proteomics = pre_process_omics(genomics,proteomics)
     
    # merge the features from omics + images in one feature vector
    omics = pd.concat([genomics, proteomics], axis=1, join="inner")
    omics_filtered = omics.loc[omics.index.isin(patient_ids)]
    omics_filtered = omics_filtered.reset_index(drop=True)

    cnn_features = pd.DataFrame(rows).set_index("Patient_ID")
    print(cnn_features)

    
    # Merge CNN and omics features:
    # Find patients present in both datasets.
    common_patients = omics.index.intersection(cnn_features.index)
    #sjpuld be 27 patients
    print(f"Common patients: {len(common_patients)}")
    omics_filtered = omics.loc[common_patients].copy()
    cnn_filtered = cnn_features.loc[common_patients].copy()

    print(omics_filtered.shape)
    print(cnn_filtered.shape)
    
    all_features = pd.concat([omics_filtered, cnn_filtered], axis=1)

    print(all_features)
    print(phenotype)

    # Train-test split (using RandomForestClassifier here instead of the NN)
    #X_train, X_test, y_train, y_test = train_test_split(all_features, phenotype, test_size=0.4, random_state=223, shuffle=True)
    cross_validation = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
    fold_acc = []
    for fold, (train_index, test_index) in enumerate(cross_validation.split(all_features, phenotype)):
        print(f"Starting fold: {fold}")
        
        X_train = all_features.iloc[train_index]
        X_test = all_features.iloc[test_index]
        y_train = phenotype.iloc[train_index]
        y_test = phenotype.iloc[test_index]
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        acc = accuracy_score(y_test, predictions)
        fold_acc.append(acc)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predictions, digits=4))

    print(f"Mean accuracy across all folds: {sum(fold_acc)/len(fold_acc):.4f}")

    # for single train-test split
    # model = RandomForestClassifier()
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    
    # print(classification_report(y_test, predictions))
    # print("Accuracy:", accuracy_score(y_test, predictions))


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
            #print(f"f{i}: {mean_feat[i]}")
        
        row["label"] = label
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Patient_ID")
    print(df)
    
    X = df.drop(columns="label")
    print(X)
    y = df["label"]
    X = torch.tensor(X.values, dtype=torch.float32)

    y = torch.tensor(phenotype, dtype=torch.long)
    print(y)

    # for single train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1, shuffle=True)

    # for k-fold cross validation
    cross_validation = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    fold_acc = []

    # Initialize model
    input_dim = 128
    num_classes = 4

    # a very shallow model for testing the extrracted features

    class ClassifierNN(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(ClassifierNN, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            return self.fc(x)
        
    for fold, (train_index, test_index) in enumerate(cross_validation.split(X, y)):
        print(f"Starting fold: {fold}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = ClassifierNN(input_dim, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001) 

        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        with torch.no_grad():
            preds = model(X_test).argmax(dim=1)
            accuracy = (preds == y_test).float().mean().item()
            fold_acc.append(accuracy)
            print(f"Test Accuracy: {accuracy:.4f}")

    print(f"Mean accuracy across all fold: {sum(fold_acc)/len(fold_acc):.4f}")

    """
    Starting fold: 0
    Epoch 10, Loss: 1.3029
    Epoch 20, Loss: 1.2329
    Epoch 30, Loss: 1.1866
    Epoch 40, Loss: 1.1703
    Epoch 50, Loss: 1.1634
    Test Accuracy: 0.5000
    Starting fold: 1
    Epoch 10, Loss: 1.3247
    Epoch 20, Loss: 1.2647
    Epoch 30, Loss: 1.2247
    Epoch 40, Loss: 1.2071
    Epoch 50, Loss: 1.1990
    Test Accuracy: 0.5385
    Mean accuracy across all fold: 0.5192
    
    """

    # model = ClassifierNN(input_dim, num_classes)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    # for epoch in range(10):
    #     optimizer.zero_grad()
    #     outputs = model(X_train)
    #     loss = criterion(outputs, y_train)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # evaluate
    # with torch.no_grad():
    #     preds = model(X_test).argmax(dim=1)
    #     accuracy = (preds == y_test).float().mean().item()
    #     print(f"Test Accuracy: {accuracy:.4f}")
    

if __name__ == "__main__":
    genomics, proteomics, clinical = load_data()
    phenotype = get_target_class(clinical)
    phenotype2 = phenotype.copy()

    #predict_cnn_only(phenotype)
    predict_prefusion(phenotype2)
