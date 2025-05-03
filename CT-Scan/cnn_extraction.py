import torch
import logging
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
import tqdm
import os
torch.cuda.empty_cache()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device selection: Try CUDA, then MPS, then fallback to CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
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

# Load the model and set up the feature extractor.
# For more robust feature representation, you can extract outputs from multiple layers.
# For example, let’s extract features from both 'avgpool' and an intermediate layer like 'layer4'.
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
resnet.eval()

# Define which layers to extract features from.
# You can change the dictionary keys/values based on which layers you find informative.
return_nodes = {
    'layer4': 'features_layer4',   # Intermediate features
    'avgpool': 'features_avgpool'    # Global pooled features
}
feature_extractor = create_feature_extractor(resnet, return_nodes=return_nodes)

# Image transformation: You can add more augmentations if needed.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Define acceptable image extensions
ACCEPTED_EXTENSIONS = {".png"}
# Set up the directory structure.
root = Path.cwd()
img_dir = root / "CT-Scan/20250423"

patient_features = {}  # Dictionary: {patient_id: list_of_feature_tensors}

# Process folders with progress feedback.
for folder in tqdm.tqdm(list(img_dir.iterdir()), desc="Processing folders"):
    # Extract patient_id from folder name (assumes format "1234_Something")
    patient_id = folder.name.split("_")[0]
    
    if patient_id not in patient_features:
        patient_features[patient_id] = {'features_layer4': [], 'features_avgpool': []}
    

    for sub_folder in folder.iterdir():
        if sub_folder.is_dir():
            for img_file in sub_folder.iterdir():
                if img_file.suffix.lower() not in ACCEPTED_EXTENSIONS:
                    logging.debug(f"Skipping non-image file: {img_file}")
                    continue

                try:
                    image = Image.open(img_file).convert("RGB")
                except (UnidentifiedImageError, OSError) as e:
                    logging.warning(f"Failed to process image {img_file}: {e}")
                    continue

                try:
                    with torch.no_grad():
                        img_tensor = transform(image).unsqueeze(0).to(device)
                        output = feature_extractor(img_tensor)
                    
                    for key, feat in output.items():
                        features = feat.view(1, -1).detach().cpu()
                        patient_features[patient_id][key].append(features)
                except Exception as e:
                    logging.error(f"Error during feature extraction for {img_file}: {e}")
                    continue
        else:
            logging.debug(f"Skipping non-directory: {sub_folder}")

aggregated_patient_features = {}


for pid, feat_dict in patient_features.items():
    aggregated = {}
    for key, feat_list in feat_dict.items():
        if feat_list:
            aggregated[key] = torch.cat(feat_list, dim=0)
        else:
            logging.warning(f"No features extracted for patient {pid} at layer {key}.")
            aggregated[key] = torch.tensor([])
    aggregated_patient_features[pid] = aggregated

root =Path("all_output")
os.makedirs(root, exist_ok=True)
torch.save(aggregated_patient_features, root/"features_23.pt")
logging.info("Saved aggregated feature tensors to 'features.pt'.")
