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
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report, accuracy_score
from sklearn.preprocessing   import StandardScaler

torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    v2.Resize(256,256),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.CenterCrop(224),
    v2.ToTensor(),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std =[0.229, 0.224, 0.225]
    )
])


def extract_all(root_dir):
    model = SimpleCNN(out_dim=128).to(device).eval()
    patient_feats = {}
    patient_labels= {}
    # use tqdm to show progress bar

    for label_dir, label_val in tqdm.tqdm([("present_png", 1), ("not_present_png", 0)]):
        for patient_folder in (root_dir/label_dir).iterdir():
            pid = patient_folder.name.split("_")[0]
            # THis will pull patient ID minus the png part
            feats = []
            for img_path in patient_folder.rglob("*.png"):
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
                patient_feats[pid]   = torch.cat(feats, dim=0)
                patient_labels[pid]  = label_val

    os.makedirs("all_output", exist_ok=True)
    torch.save(patient_feats,  "all_output/custom_feats.pt")
    torch.save(patient_labels, "all_output/labels.pt")
    print(f"Extracted features for {len(patient_feats)} patients.")


def main():
    root_dir = Path.cwd()

    print(f"Current working directory: {root_dir}")
    print("Extracting features...")
    extract_all(root_dir/"CT-Scan")

if __name__ == "__main__":
    main()


# feats_dict  = torch.load("all_output/custom_feats.pt")  
# labels_dict = torch.load("all_output/labels.pt")        

# # 2. Build a DataFrame with one row per patient
# rows = []
# for pid, feats in feats_dict.items():
#     mean_feat = feats.mean(dim=0).numpy()  # [128]
#     label     = labels_dict[pid]
#     rows.append({"Patient_ID": pid, **{f"f{i}": mean_feat[i] for i in range(mean_feat.shape[0])}, "label": label})

# df = pd.DataFrame(rows).set_index("Patient_ID")

# # 3. Features / labels
# X = df.drop(columns="label")
# y = df["label"]

# # 4. Scale, split, train
# scaler = StandardScaler()
# Xs = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     Xs, y, test_size=0.3, random_state=42, stratify=y
# )

# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
# pred = clf.predict(X_test)

# # 5. Evaluate
# print("Accuracy:", accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))
