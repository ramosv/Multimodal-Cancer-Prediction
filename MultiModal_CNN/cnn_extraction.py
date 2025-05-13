import torch
import logging
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms import v2
import tqdm
import torch.nn as nn
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import sys

torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream = sys.stdout)

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps"  if torch.backends.mps.is_available() else 
                      "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc   = nn.Linear(64, out_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize( 
        mean=[0.485, 0.456, 0.406], 
        std =[0.229, 0.224, 0.225]
    )
])

transform2 = v2.Compose([
    v2.Resize(256),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.CenterCrop(224),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std =[0.229, 0.224, 0.225]
    )
    ])

def extract_all(root_dir, timestamp):
    model = SimpleCNN(out_dim=128).to(device).eval()
    patient_feats = {}
    patient_labels= {}
    # use tqdm to show progress bar
    # move tqdm to inner loop
    present_patients = []
    for patient_folder in (root_dir/"present").iterdir():
        pid = patient_folder.name
        present_patients.append(pid)

    not_present_patients = []
    for patient_folder in (root_dir/"not_present").iterdir():
        pid = patient_folder.name
        not_present_patients.append(pid)

    logging.info(f"Present patients: {len(present_patients)}")
    logging.info(f"Not present patients: {len(not_present_patients)}")

    for png_dir in Path(root_dir/timestamp).iterdir():
        pid = png_dir.name.split("_")[0]
        for patient_folder in png_dir.iterdir():
            # THis will pull patient ID minus the png part
            feats = []
            logging.info(f"Extracting Features from patient: {pid}")
            for img_path in tqdm.tqdm(patient_folder.rglob("*.png")):
                try:
                    img = Image.open(img_path).convert("RGB")
                    with torch.no_grad():
                        #t = transform(img).unsqueeze(0).to(device)
                        t = transform2(img).unsqueeze(0).to(device)
                        f = model(t)
                    feats.append(f.cpu())

                except (UnidentifiedImageError, OSError):
                    continue
            if feats:
                patient_feats[pid] = torch.cat(feats, dim=0)
                if pid in present_patients:
                    patient_labels[pid] = 1
                elif pid in not_present_patients:
                    patient_labels[pid] = 0
                else:
                    logging.warning(f"Patient {pid} not found in present or not present patients")
                    continue

    os.makedirs("all_output", exist_ok=True)
    torch.save(patient_feats, "all_output/custom_feats.pt")
    torch.save(patient_labels, "all_output/labels.pt")
    print(f"Extracted features for {len(patient_feats)} patients.")

def freezing_cnn(root_dir, timestamp):
    # Puts layers in inference mode
    model = SimpleCNN(out_dim=128).to(device).eval()
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    patient_feats = {}
    patient_labels= {}

    present_patients = []
    for patient_folder in (root_dir/"present").iterdir():
        pid = patient_folder.name
        present_patients.append(pid)

    not_present_patients = []
    for patient_folder in (root_dir/"not_present").iterdir():
        pid = patient_folder.name
        not_present_patients.append(pid)

    logging.info(f"Present patients: {len(present_patients)}")
    logging.info(f"Not present patients: {len(not_present_patients)}")

    # use tqdm to show progress bar
    #move tqdm to inner loop

    for png_dir in Path(root_dir/timestamp).iterdir():
        pid = png_dir.name.split("_")[0]
        for patient_folder in png_dir.iterdir():
            # THis will pull patient ID minus the png part
            feats = []
            for img_path in tqdm.tqdm(patient_folder.rglob("*.png")):
                try:
                    img = Image.open(img_path).convert("RGB")
                    with torch.no_grad():
                        #t = transform(img).unsqueeze(0).to(device)
                        t = transform2(img).unsqueeze(0).to(device)
                        f = model(t)
                    feats.append(f.cpu())
                except (UnidentifiedImageError, OSError):
                    continue

            if feats:
                patient_feats[pid] = torch.cat(feats, dim=0)
                if pid in present_patients:
                    patient_labels[pid] = 1
                elif pid in not_present_patients:
                    patient_labels[pid] = 0
                else:
                    logging.warning(f"Patient {pid} not found in present or not present patients")
                    continue

    os.makedirs("all_output", exist_ok=True)
    torch.save(patient_feats, "all_output/custom_feats_frozen.pt")
    torch.save(patient_labels, "all_output/labels_frozen.pt")
    print(f"Frozen features for {len(patient_feats)} patients.")

def run_predictions():
    feats_dict  = torch.load("all_output/custom_feats.pt")  
    labels_dict = torch.load("all_output/labels.pt")

    # our labels_dict is a dictionary with patient id and value is either 0 or 1
    # example: {'C3L-00610': 1, 'C3L-00609': 0, ...}        

    for key, value in labels_dict.items():
        print(f"{key}: {value}")


    for key, value in feats_dict.items():
        print(f"{key}: {value}")

    rows = []
    for pid, feats in feats_dict.items():
        mean_feat = feats.mean(dim=0).numpy()
        label = labels_dict[pid]
        row = {"Patient_ID": pid}
        for i in range(mean_feat.shape[0]):
            row[f"f{i}"] = mean_feat[i]
            #print(f"f{i}: {mean_feat[i]}")
        
        row["label"] = label
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Patient_ID")

    X = df.drop(columns="label")
    y = df["label"]

    # 4. Scale, split, train
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

#with transform version 1
"""
Accuracy: 0.4
              precision    recall  f1-score   support

           0       0.50      0.58      0.54        12
           1       0.17      0.12      0.14         8

    accuracy                           0.40        20
   macro avg       0.33      0.35      0.34        20
weighted avg       0.37      0.40      0.38        20

# Transform version 2
Accuracy: 0.47368421052631576
              precision    recall  f1-score   support

           0       0.54      0.64      0.58        11
           1       0.33      0.25      0.29         8

    accuracy                           0.47        19
   macro avg       0.44      0.44      0.43        19
weighted avg       0.45      0.47      0.46        19

"""

# def main():
#     root_dir = Path.cwd()
    
#     print(f"Current working directory: {root_dir}")
#     print("Extracting features...")
#     extract_all(root_dir/"CT-Scan")
#     run_predictions()
#     freezing_cnn(root_dir/"CT-Scan")

# if __name__ == "__main__":
#     main()


