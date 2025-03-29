import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from pathlib import Path

resnet = models.resnet18(pretrained=True)
resnet.eval()

return_nodes = {'avgpool':'features'}
feature_extractor = create_feature_extractor(resnet,return_nodes=return_nodes)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),               
    transforms.Normalize(                 
        mean=[0.485, 0.456, 0.406],       
        std=[0.229, 0.224, 0.225]
    )
])

img_dir = Path(r"C:\Users\ramos\Desktop\GitHub\Clear-Cell-Carcinoma-Study\CT-Scan\20250329")
all_tensors = []
for folder in img_dir.iterdir():
    for sub_folder in folder.iterdir():
        if sub_folder.is_dir():
            for img in sub_folder:
                image = Image.open(img).convert("RGB")
                img_tensor = transform(image).unsqueeze(0)
                all_tensors.append(img_tensor)
        else:
            print(f"Skipping non-directory: {sub_folder}")

batch_tensor = torch.stack(all_tensors, dim=0)

with torch.no_grad():
    output = feature_extractor(batch_tensor)
    features = output['features']

features = features.view(features.size(0), -1)

print(features.shape) 