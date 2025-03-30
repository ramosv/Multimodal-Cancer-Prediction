import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from pathlib import Path

resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.eval()

return_nodes = {'avgpool': 'features'}
feature_extractor = create_feature_extractor(resnet, return_nodes=return_nodes)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

root = Path.cwd()
img_dir = root / "CT-Scan/20250329"

patient_features = {}  # Dictionary: {patient_id: list_of_feature_tensors}

for folder in img_dir.iterdir():
    # Example folder name: "1234_Something" so patient_id = "1234"
    parts = folder.name.split("_")
    patient_id = parts[0]

    # Ensure we have a list ready for this patient
    if patient_id not in patient_features:
        patient_features[patient_id] = []

    for sub_folder in folder.iterdir():
        if sub_folder.is_dir():
            for img in sub_folder.iterdir():
                image = Image.open(img).convert("RGB")
                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = feature_extractor(img_tensor)
                features = output['features'].view(1, -1)  # shape [1, 512] for ResNet18

                # Collect the extracted feature for this patient
                patient_features[patient_id].append(features)
        else:
            print(f"Skipping non-directory: {sub_folder}")

# At this point, patient_features[patient_id] is a list of [1, feature_dim] tensors
# Optionally stack/aggregate per patient. For example:
for pid, feats in patient_features.items():
    patient_features[pid] = torch.cat(feats, dim=0)  # shape [num_images_for_that_pid, 512]

# Finally, save the dictionary
torch.save(patient_features, "features.pt")

print("Saved dictionary of patient -> feature_tensor.")