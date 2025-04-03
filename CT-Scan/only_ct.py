from pathlib import Path
import shutil

root = Path.cwd()
img_dir = root / "CT-Scan/20250329"

for folder in img_dir.iterdir():
    for subfoler in folder.iterdir():
        folder_name = subfoler.name
        if "CT" not in folder_name:
            print(folder_name)
