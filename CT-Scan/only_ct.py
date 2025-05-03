from pathlib import Path
import shutil

root = Path.cwd()
img_dir = root / "CT-Scan/20250423"

for folder in img_dir.iterdir():
    for subfoler in folder.iterdir():
        folder_name = subfoler.name
        if "CT" not in folder_name:
            print(f"{subfoler}_{folder_name}")
            #remove folder
            #shutil.rmtree(subfoler)


