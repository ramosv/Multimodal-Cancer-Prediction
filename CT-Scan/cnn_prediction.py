import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split

class ClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassifierNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

def predict(phenotype):
    num_classes = 2
    features_dict = torch.load("features.pt")  # Expected: Dict[Patient_ID] = feature tensor
    patient_ids = list(features_dict.keys())
    phenotype = phenotype.squeeze()  # 'phenotype' is a Series/DataFrame indexed by patient_id

    X, y = [], []
    for pid in patient_ids:
        if pid in phenotype.index:
            avg_features = features_dict[pid].mean(dim=0)  # shape [512]
            X.append(avg_features.numpy())

            # If phenotype is truly a Series, this returns a single value:
            y.append(phenotype.loc[pid])

    X = torch.tensor(X, dtype=torch.float32)  # shape [N, 512]
    y = torch.tensor(y, dtype=torch.long)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=223, shuffle=True)

    # Initialize model
    input_dim = X_train.shape[1]  # Feature dimension from CNN
    num_classes = 2  # Assuming binary classification: Stage 1 vs. Stage 2

    model = ClassifierNN(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):
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

def load_data():
    root = Path("jessica_output")
    genomics = pd.read_csv(root / "genes_filtered.csv", index_col=0)
    proteomics = pd.read_csv(root / "proteins_filtered.csv", index_col=0)
    clinical = pd.read_csv(root / "clinical_filtered.csv",index_col=0)
    return genomics, proteomics, clinical

def get_target_class(clinical):
    present_patients = Path("CT-Scan/present")
    relevant_patients =[]
    for folder in present_patients.iterdir():
        relevant_patients.append(folder.name)
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
    # print(phenotype.value_counts(sort=True))
    return phenotype

if __name__ == "__main__":
    genomics, proteomics, clinical = load_data()
    phenotype = get_target_class(clinical)
    predict(phenotype)
